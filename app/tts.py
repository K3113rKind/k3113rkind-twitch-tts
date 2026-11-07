"""Kokoro-TTS – Sprachsynthese anhand eines Stimmen-Schlüssels.

Vereinfachte Fassung: keine Engine-/Modellauswahl nach außen. Der Aufrufer
übergibt nur einen Schlüssel aus voices.VOICES; Backbone, Voicepack und
Sprachcode ergeben sich daraus.

Benötigt die semidark-Forks von `kokoro` (lang_code "d" für Deutsch,
upstream nur als offener PR hexgrad/kokoro#317) und `misaki[de]` (DEG2P) –
siehe requirements.txt. Mit den PyPI-Standardpaketen erzeugen die
kikiri-Checkpoints nur Rauschen.

Die Synthese delegiert vollständig an KPipeline (G2P, Chunking,
Style-Vector-Indizierung), wie im Referenzskript von semidark/kikiri-tts.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import threading
import time

from . import voices as voices_mod

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
REPO_ID = "hexgrad/Kokoro-82M"  # Architektur-Referenz; Gewichte kommen lokal


class SynthesisResult:
    """Rohe Samples (float32, mono, SAMPLE_RATE) – kodiert wird erst im
    Audio-Stream (stream.py)."""

    __slots__ = ("samples", "duration_seconds", "sample_rate", "compute_seconds",
                 "load_seconds", "cached", "decode_seconds", "batch")

    def __init__(self, samples, duration_seconds: float, sample_rate: int,
                 compute_seconds: float = 0.0, load_seconds: float = 0.0,
                 cached: bool = False, decode_seconds: float = 0.0, batch: int = 1):
        self.samples = samples
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.compute_seconds = compute_seconds  # reine Sprachberechnung
        self.load_seconds = load_seconds        # Modell laden (nur beim ersten Mal)
        self.cached = cached                    # aus dem Zwischenspeicher, nicht neu berechnet
        self.decode_seconds = decode_seconds    # davon Klangerzeugung (Qwen-Decoder)
        self.batch = batch                      # mit so vielen Nachrichten gemeinsam berechnet


def _resample(samples, sr_in: int, sr_out: int):
    """Lineare Umrechnung – reicht für Sprache, vermeidet weitere Pakete."""
    import numpy as np

    if sr_in == sr_out or len(samples) == 0:
        return samples
    n_out = int(round(len(samples) * sr_out / sr_in))
    x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


# Qwen erzeugt 12,5 Audio-Schritte pro Sekunde und hört erst auf, wenn es
# das Satzende selbst erkennt – sonst bis 2048 Schritte (~2,7 min Audio,
# auf CPU über 10 min Rechenzeit). Passierte z. B. mit int8-Quantisierung (verworfen).
# Obergrenze: sehr langsames Sprechtempo (6 Zeichen/s) plus
# 5 s Puffer – normale Sätze erreichen sie nie.
QWEN_STEPS_PER_SECOND = 12.5


def _qwen_token_budget(text: str) -> int:
    seconds = len(text) / 6 + 5
    return min(2048, int(seconds * QWEN_STEPS_PER_SECOND))


# Längere Texte rechnet Qwen abschnittsweise: Pro Durchgang schafft es
# höchstens 2048 Schritte (~2,7 min Audio), und kürzere Abschnitte halten
# das Modell stabil. 300 Zeichen sind gut 20 s Sprache – weit unter dem
# Limit. Kokoro teilt selbst auf und braucht das nicht.
QWEN_CHUNK_CHARS = 300
QWEN_CHUNK_PAUSE_S = 0.25


# Der erste Abschnitt eines langen Texts ist kürzer, damit der Ton schneller
# startet; die übrigen rechnen, während er läuft. Danach wachsen die
# Abschnitte schrittweise (x1,4): Ein Abschnitt ist dann fertig, bevor der
# vorherige zu Ende gesprochen ist (bei RTF bis ~0,7) – keine Pause nach dem
# ersten Satz. Ein großer Sprung von 120 auf 300 Zeichen verursachte eine.
QWEN_FIRST_CHUNK_CHARS = 120
QWEN_CHUNK_GROWTH = 1.4


def split_for_qwen(text: str, max_chars: int = QWEN_CHUNK_CHARS,
                   first_max: int | None = None) -> list[str]:
    """Text in Abschnitte aus GANZEN Sätzen teilen – ein Satz wird nie
    zerschnitten, egal wie lang er ist (ein Wechsel mitten im Satz klingt
    unnatürlich). Kurze Sätze werden zusammengefasst, solange der Abschnitt
    max_chars nicht überschreitet; der erste Abschnitt bis first_max (für
    einen schnellen Start), danach wachsen die Abschnitte."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    sentences = _sentences(text)
    chunks: list[str] = []
    limit = first_max or max_chars
    for sentence in sentences:
        if chunks and len(chunks[-1]) + 1 + len(sentence) <= limit:
            chunks[-1] += " " + sentence
            continue
        if chunks:
            # nächster Abschnitt darf größer werden (Vorlauf wächst mit)
            limit = min(max_chars, max(int(limit * 1.2), int(len(chunks[-1]) * QWEN_CHUNK_GROWTH)))
        chunks.append(sentence)
    # winzigen Rest an den vorletzten Abschnitt hängen
    if len(chunks) > 1 and len(chunks[-1]) < 40:
        tail = chunks.pop()
        chunks[-1] += " " + tail
    return chunks


# Punkt ist kein Satzende hinter einer Zahl (Ordnungszahl/Datum: "am 7.
# November", "der 3. Platz", "07.11.") oder hinter einer Abkürzung.
_ABBREV = ("z. B", "z.B", "d. h", "d.h", "u. a", "u.a", "o. Ä", "u. U", "bzw", "ca", "evtl",
           "ggf", "inkl", "exkl", "vgl", "Nr", "Dr", "Prof", "Str", "mind", "max", "min", "etc",
           "sog", "bzgl", "bspw", "usw", "St", "Mio", "Mrd", "Tsd", "Jh", "Abs", "Kap", "Std", "Min",
           "Sek", "Hr", "Fr", "geb", "gest", "ugs", "allg", "zzgl", "Tel")
_NO_SENTENCE_END = re.compile(
    r"(?:(?<![\w])\d+\.?|(?<![\w'’])[^\W\d_]\.?"  # Zahl oder einzelner Buchstabe (z. B.)
    r"|(?<![\w])(?:" + "|".join(re.escape(a) for a in _ABBREV) + r")\.?)$")
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"“”»«'’)\]]*\s+|\n+")


def _sentences(text: str) -> list[str]:
    """Text in ganze Sätze teilen (nur an . ! ? … und Zeilenumbrüchen –
    nie an Komma, Semikolon oder Doppelpunkt). Anführungszeichen/Klammern
    hinter dem Satzende bleiben beim Satz."""
    parts, pos = [], 0
    for m in _SENTENCE_END.finditer(text):
        end = m.start() + len(m.group(0).rstrip())
        if "\n" not in m.group(0) and _NO_SENTENCE_END.search(text[pos:m.start()]):
            continue  # "7. November", "1. FC", "z. B.", "Nr." – kein Satzende
        piece = text[pos:end].strip()
        if piece:
            parts.append(piece)
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        parts.append(tail)
    return parts


# Qwen würfelt bei jeder Erzeugung (Sampling, Temperatur 0.9) – derselbe Satz
# klang deshalb jedes Mal anders betont. Mit festem Startwert vor jedem Satz
# ist das Ergebnis für denselben Text und dieselbe Stimme immer gleich;
# verschiedene Sätze klingen trotzdem natürlich unterschiedlich.
# Qwen verschluckt am Ende gern den letzten Laut. Kniff aus der
# Qwen3-TTS-Community: Füllzeichen hinter das Satzende. Auf Deutsch erzeugt
# das aber gelegentlich Fantasie-Silben ("plili") -> standardmäßig aus,
# QWEN_END_FIX=1 schaltet es ein.
QWEN_END_FIX = os.environ.get("QWEN_END_FIX", "0").strip() not in ("0", "false", "no", "")
QWEN_END_TAIL = " ^."


def _qwen_text(chunk: str) -> str:
    chunk = chunk.rstrip()
    if not chunk:
        return chunk
    if chunk[-1] not in ".!?…:":  # ":" = Namensansage, Stimme bleibt "offen"
        chunk += "."
    return chunk + QWEN_END_TAIL if QWEN_END_FIX else chunk


# Schnelles Klonen: nur der Stimm-Fingerabdruck (x-vector) statt der ganzen
# Referenzaufnahme als Kontext. Kürzerer Kontext -> schneller, aber die
# Stimme trifft weniger genau. QWEN_CLONE_FAST=1 schaltet es ein.
QWEN_CLONE_FAST = os.environ.get("QWEN_CLONE_FAST", "0").strip() in ("1", "true", "yes")

# Stimm-Prüfung für eigene Stimmen: unter dieser Ähnlichkeit zur Vorlage
# wird neu erzeugt (0 = Prüfung aus), höchstens QWEN_SPK_RETRIES Mal.
QWEN_SPK_MIN = float(os.environ.get("QWEN_SPK_MIN", "0.75") or 0)
QWEN_SPK_RETRIES = int(os.environ.get("QWEN_SPK_RETRIES", "2") or 0)

QWEN_SEED = int(os.environ.get("QWEN_SEED", "1234") or 1234)

# Fertige Qwen-Ausgaben merken: Ein wiederholter Text (Testsatz, "Hallo")
# kommt dann sofort statt nach 10–20 s Rechenzeit.
QWEN_CACHE_SIZE = 64

# Dauerhafter Zwischenspeicher auf der Platte (übersteht Neustarts): fertige
# Sätze/Namensansagen der neuronalen Stimmen. Schlüssel = Stimme + Referenz-
# Änderungszeit + Einstellungen + Text -> neu geklonte Stimme oder geänderte
# Betonung erzeugen automatisch neue Einträge. Älteste fliegen raus, wenn
# TTS_CACHE_MAX_MB überschritten wird. 0 = aus.
DISK_CACHE_DIR = Path(os.environ.get(
    "TTS_CACHE_DIR",
    str(Path(os.environ.get("CONFIG_PATH", "/config/config.json")).parent / "tts-cache")))
DISK_CACHE_MAX_MB = int(os.environ.get("TTS_CACHE_MAX_MB", "2048") or 0)


class DiskCache:
    def __init__(self, folder: Path, max_mb: int) -> None:
        self.folder, self.max_bytes = folder, max_mb * 2**20
        self._puts = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_bytes > 0

    def _path(self, key: tuple) -> Path:
        import hashlib

        h = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()
        return self.folder / h[:2] / f"{h}.npy"

    def has(self, key: tuple) -> bool:
        return self.enabled and self._path(key).is_file()

    def get(self, key: tuple):
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            import numpy as np

            data = np.load(path)
            os.utime(path)  # zuletzt benutzt -> fliegt als Letztes raus
            return data.astype(np.float32) / 32767.0
        except FileNotFoundError:
            return None
        except Exception as exc:  # kaputte Datei (z. B. Absturz beim Schreiben) -> neu rechnen
            log.warning("Zwischenspeicher: %s unlesbar (%s) – wird neu berechnet", path.name, exc)
            path.unlink(missing_ok=True)
            return None

    def put(self, key: tuple, samples) -> None:
        if not self.enabled or samples is None or len(samples) == 0:
            return
        import numpy as np

        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # eigener Name je Thread: gleicher Text kann parallel fertig werden
            tmp = path.with_name(f"{path.stem}.{threading.get_ident()}.tmp")
            with open(tmp, "wb") as fh:
                np.save(fh, (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16))
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("Zwischenspeicher: Schreiben fehlgeschlagen: %s", exc)
            return
        with self._lock:
            self._puts += 1
            if self._puts % 50 == 0:
                try:  # Aufräumen darf das Ergebnis nie kosten
                    self._prune()
                except Exception as exc:
                    log.warning("Zwischenspeicher: Aufräumen fehlgeschlagen: %s", exc)

    def _prune(self) -> None:
        files = []
        for f in self.folder.glob("*/*.npy"):
            try:
                st = f.stat()
            except OSError:
                continue  # gerade ersetzt/gelöscht
            files.append((st.st_mtime, st.st_size, f))
        total = sum(size for _, size, _ in files)
        if total <= self.max_bytes:
            return
        for _, size, f in sorted(files):
            f.unlink(missing_ok=True)
            total -= size
            if total <= self.max_bytes * 0.9:
                break
        log.info("Zwischenspeicher aufgeräumt: %.0f MB", total / 2**20)

# Eigene Stimmen: Qwen wandelt bei jedem Satz die komplette Referenz (~10 s)
# erneut mit in Ton um und schneidet sie danach ab – nur für einen sauberen
# Übergang. Gemessen kostete das ein Drittel der Rechenzeit. Als Übergang
# reichen die letzten Sekunden; der Rest wird durch Stille gleicher Länge
# ersetzt, damit Qwens eigener Schnitt an derselben Stelle landet.
# QWEN_REF_DECODE_S=0 schaltet die Kürzung ab (volle Referenz wie bisher).
QWEN_REF_DECODE_S = float(os.environ.get("QWEN_REF_DECODE_S", "2") or 2)


def _qwen_device():
    """GPU nutzen, wenn eine da ist (NVIDIA per CUDA oder AMD per ROCm – beide
    erscheinen in Torch als "cuda"), sonst CPU. QWEN_DEVICE=cpu erzwingt CPU."""
    import torch

    wanted = os.environ.get("QWEN_DEVICE", "auto").strip().lower()
    if wanted != "cpu" and torch.cuda.is_available():
        if torch.version.hip and os.environ.get("QWEN_MIOPEN", "0") != "1":
            # AMD: MIOpen sucht für jede neue Eingabelänge erst einen Rechenweg
            # und landete dabei auf einem langsamen Notweg – der Klang-Decoder
            # brauchte so 65 s für 9 s Audio, der Desktop fror ein. PyTorchs
            # eigene GPU-Faltungen umgehen das. QWEN_MIOPEN=1 schaltet es zurück.
            torch.backends.cudnn.enabled = False
            log.info("ROCm: MIOpen für Faltungen abgeschaltet (QWEN_MIOPEN=1 zum Testen)")
        # Genauigkeit auf der GPU (QWEN_GPU_DTYPE): Standard float32 wie auf
        # der CPU. bfloat16 spart Speicher, klang aber hörbar anders (Betonung,
        # teils abgeschnittene Satzenden); da Qwen hier ohnehin vom Verwaltungs-
        # aufwand gebremst wird, kostet float32 kaum Tempo.
        dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        name = os.environ.get("QWEN_GPU_DTYPE", "float32").strip().lower()
        dtype = dtypes.get(name, torch.float32)
        return "cuda:0", dtype, torch.cuda.get_device_name(0)
    return "cpu", torch.float32, "CPU"


# Experiment (nur GPU): torch.compile fasst die vielen kleinen Rechenschritte
# von Qwen zu größeren zusammen. QWEN_COMPILE: 0/aus (Standard), 1 = normal,
# overhead = "reduce-overhead" (GPU-Graphen), max = "max-autotune-no-cudagraphs".
# Erster Satz nach dem Start kompiliert (Minuten); scheitert es, schaltet die
# App für diesen Lauf automatisch zurück auf ohne.
_COMPILE_MODES = {"1": "default", "on": "default", "overhead": "reduce-overhead",
                  "max": "max-autotune-no-cudagraphs"}


def _compile_mode() -> str | None:
    return _COMPILE_MODES.get(os.environ.get("QWEN_COMPILE", "0").strip().lower())


class QwenEngine:
    """Qwen3-TTS auf CPU – Preset-Stimmen (CustomVoice-Modelle) und eigene
    Stimmen (Base-Modelle mit Referenzaufnahme). Hält nur ein Modell im RAM –
    beim Wechsel wird das vorherige freigegeben.

    Stellschraube QWEN_THREADS (Umgebungsvariable, .env): Anzahl
    Torch-Threads; 0 = alle sichtbaren Kerne. Oft schneller: nur echte Kerne
    eines CPU-Sockels statt Hyper-Threads über beide Sockel verteilt.

    int8-Quantisierung wurde getestet und verworfen: Auf dem Xeon Gold 6150
    (Skylake, ohne VNNI) fand das Modell damit das Satzende nicht mehr.
    """

    def __init__(self) -> None:
        self._repo: str | None = None
        self._model = None
        self._lock = threading.Lock()
        # Vorberechnete Referenzen fürs Klonen: (Modell, wav, Änderungszeit)
        self._clone_prompts: dict[tuple, object] = {}
        self._decode_s = 0.0
        self.last_decode_s = 0.0
        self._trim_codes = 0  # beim Klonen: so viele Referenz-Schritte nicht decodieren
        self.device_name = None  # "CPU" oder Name der Grafikkarte, sobald geladen
        self._compiled: list = []  # (Modul, ursprüngliches forward) bei torch.compile

    def _uncompile(self, exc: Exception) -> None:
        log.warning("torch.compile fehlgeschlagen (%s) – rechne ab jetzt ohne", exc)
        for mod, original in self._compiled:
            mod.forward = original
        self._compiled = []
        if self.device_name:
            self.device_name = self.device_name.split(" · compile")[0] + " · compile fehlgeschlagen"

    @staticmethod
    def _decode_trimmed(original_decode, encoded, trim, *args, **kwargs):
        """Nur das Ende der Referenz + die neue Ausgabe decodieren und vorne
        mit Stille auf die volle Länge auffüllen."""
        import numpy as np

        totals = [len(item["audio_codes"]) for item in encoded]
        shortened = [{**item, "audio_codes": item["audio_codes"][trim:]} for item in encoded]
        wavs, fs = original_decode(shortened, *args, **kwargs)
        padded = []
        for wav, total in zip(wavs, totals):
            kept = max(total - trim, 1)
            pad = int(round(len(wav) * trim / kept))
            padded.append(np.concatenate([np.zeros(pad, dtype=wav.dtype), wav]))
        return padded, fs

    def load(self, repo_id: str) -> None:
        with self._lock:
            if self._repo == repo_id:
                return
            snapshot = voices_mod.hf_snapshot(repo_id)
            if snapshot is None:
                raise FileNotFoundError(
                    f"Qwen-Modell {repo_id} nicht im Hugging-Face-Cache gefunden (HF_HOME)."
                )
            import gc

            import torch
            from qwen_tts import Qwen3TTSModel

            self._model = None
            self._clone_prompts.clear()
            gc.collect()
            threads = int(os.environ.get("QWEN_THREADS", "0") or 0)
            if threads > 0:
                torch.set_num_threads(threads)
            device, dtype, device_name = _qwen_device()
            log.info(
                "Lade Qwen3-TTS: %s auf %s (%s)", repo_id, device_name,
                f"{torch.get_num_threads()} Threads" if device == "cpu" else str(dtype),
            )
            model = Qwen3TTSModel.from_pretrained(str(snapshot), device_map=device, dtype=dtype)
            self.device_name = device_name if device == "cpu" else (
                f"{device_name} · {str(dtype).replace('torch.', '')}"
            )
            self._compiled = []
            mode = _compile_mode()
            if mode and device != "cpu":
                # Nur die reinen Transformer-Blöcke kompilieren – die
                # Generierungsschleifen drumherum bleiben normales Python.
                talker = model.model.talker
                for mod in (talker.model, talker.code_predictor.model):
                    original = mod.forward
                    mod.forward = torch.compile(original, mode=mode, dynamic=True)
                    self._compiled.append((mod, original))
                self.device_name = f"{self.device_name} · compile {mode}"
                log.info("torch.compile aktiv (%s) – erster Satz dauert länger", mode)
            # Zeit des Klang-Decoders mitmessen (Diagnose: Anteil an der Rechenzeit)
            tokenizer = model.model.speech_tokenizer
            original_decode = tokenizer.decode

            def timed_decode(encoded, *args, **kwargs):
                t = time.perf_counter()
                try:
                    trim = self._trim_codes
                    if trim > 0 and isinstance(encoded, list):
                        return self._decode_trimmed(original_decode, encoded, trim, *args, **kwargs)
                    return original_decode(encoded, *args, **kwargs)
                finally:
                    self._decode_s += time.perf_counter() - t

            tokenizer.decode = timed_decode
            self._model = model
            self._repo = repo_id

    def _speaker_similarity(self, item, wav, sr: int) -> float | None:
        """Kosinus-Ähnlichkeit zwischen Stimm-Fingerabdruck der Vorlage und
        dem erzeugten Audio (Sprecher-Encoder des Qwen-Modells, 24 kHz)."""
        try:
            import torch

            ref = getattr(item, "ref_spk_embedding", None)
            if ref is None or len(wav) < sr // 4:
                return None
            audio = _resample(wav.astype("float32"), sr, 24000)
            with torch.inference_mode():
                emb = self._model.model.extract_speaker_embedding(audio=audio, sr=24000)
                a = emb.float().flatten()
                b = ref.to(emb.device).float().flatten()
                return float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        except Exception as exc:  # Prüfung darf die Ausgabe nie verhindern
            log.warning("Stimm-Prüfung nicht möglich: %s", exc)
            return None

    def _clone_prompt(self, ref_wav: str, ref_txt: str):
        """Referenz einmal vorberechnen (spart das bei jedem Satz)."""
        from pathlib import Path

        key = (self._repo, ref_wav, Path(ref_wav).stat().st_mtime, Path(ref_txt).stat().st_mtime,
               QWEN_CLONE_FAST)
        if key not in self._clone_prompts:
            text = Path(ref_txt).read_text(encoding="utf-8").strip()
            t0 = time.perf_counter()
            self._clone_prompts[key] = self._model.create_voice_clone_prompt(
                ref_audio=ref_wav, ref_text=text, x_vector_only_mode=QWEN_CLONE_FAST
            )
            log.info("Referenz %s vorbereitet in %.1f s%s", Path(ref_wav).name,
                     time.perf_counter() - t0, " (schnell: nur Stimm-Fingerabdruck)" if QWEN_CLONE_FAST else "")
        return self._clone_prompts[key]

    def synthesize(self, text: str, repo_id: str, speaker: str | None = None,
                   ref_wav: str | None = None, ref_txt: str | None = None,
                   temperature: float | None = None, instruct: str | None = None):
        import numpy as np
        import torch

        self.load(repo_id)
        chunks = split_for_qwen(text)
        parts = []
        self._decode_s = 0.0
        t0 = time.perf_counter()
        label = speaker or (ref_wav and ref_wav.rsplit("/", 1)[-1].removesuffix(".wav"))
        for i, chunk in enumerate(chunks, 1):
            with self._lock:
                trim = 0
                prompt = None
                if ref_wav:
                    prompt = self._clone_prompt(ref_wav, ref_txt)
                    ref_code = getattr(prompt[0], "ref_code", None) if prompt else None
                    keep = int(QWEN_REF_DECODE_S * QWEN_STEPS_PER_SECOND)
                    if QWEN_REF_DECODE_S > 0 and ref_code is not None:
                        trim = max(0, int(ref_code.shape[0]) - keep)
                    gen = lambda: self._model.generate_voice_clone(  # noqa: E731
                        text=_qwen_text(chunk), language="German",
                        voice_clone_prompt=prompt,
                        max_new_tokens=_qwen_token_budget(chunk),
                        temperature=temperature,
                    )
                else:
                    gen = lambda: self._model.generate_custom_voice(  # noqa: E731
                        text=_qwen_text(chunk), language="German", speaker=speaker,
                        max_new_tokens=_qwen_token_budget(chunk),
                        temperature=temperature,
                        instruct=instruct or None,
                    )

                def run(seed: int):
                    self._trim_codes = trim
                    try:
                        torch.manual_seed(seed)
                        try:
                            return gen()
                        except Exception as exc:
                            if not self._compiled:
                                raise
                            self._uncompile(exc)
                            torch.manual_seed(seed)
                            return gen()
                    finally:
                        self._trim_codes = 0

                wavs, sr = run(QWEN_SEED)
                # Eigene Stimme: prüfen, ob das Ergebnis nach der Vorlage klingt.
                # Das Base-Modell greift selten komplett daneben (andere Stimme,
                # z. B. männlich) – dann mit anderem Startwert neu würfeln.
                if prompt and QWEN_SPK_MIN > 0:
                    sim = self._speaker_similarity(prompt[0], wavs[0], int(sr))
                    best = (sim, wavs, sr)
                    attempt = 0
                    while sim is not None and sim < QWEN_SPK_MIN and attempt < QWEN_SPK_RETRIES:
                        attempt += 1
                        log.warning("Abschnitt %d/%d klingt nicht nach der Vorlage (Ähnlichkeit %.2f "
                                    "< %.2f) – neuer Versuch %d", i, len(chunks), sim, QWEN_SPK_MIN, attempt)
                        wavs, sr = run(QWEN_SEED + attempt)
                        sim = self._speaker_similarity(prompt[0], wavs[0], int(sr))
                        if sim is not None and sim > best[0]:
                            best = (sim, wavs, sr)
                    sim, wavs, sr = best
                    if sim is not None:
                        log.info("Abschnitt %d/%d: Stimm-Ähnlichkeit %.2f", i, len(chunks), sim)
            audio = _resample(wavs[0].astype("float32"), int(sr), SAMPLE_RATE)
            # 0,15 s Stille anhängen: kein hartes Ende direkt nach dem letzten Laut
            audio = np.concatenate([audio, np.zeros(int(0.15 * SAMPLE_RATE), dtype=np.float32)])
            dur = len(audio) / SAMPLE_RATE
            log.info("Abschnitt %d/%d: %d Zeichen -> %.1f s Audio, Ende: »…%s«",
                     i, len(chunks), len(chunk), dur, chunk[-40:])
            if dur >= _qwen_token_budget(chunk) / QWEN_STEPS_PER_SECOND - 0.5:
                log.warning(
                    "Qwen hat das Satzende nicht gefunden, Abschnitt %d/%d nach %.0f s "
                    "Audio abgeschnitten.", i, len(chunks), dur,
                )
            if parts:
                parts.append(np.zeros(int(QWEN_CHUNK_PAUSE_S * SAMPLE_RATE), dtype=np.float32))
            parts.append(audio)
        took = time.perf_counter() - t0
        samples = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        dur = len(samples) / SAMPLE_RATE
        self.last_decode_s = self._decode_s
        log.info(
            "Qwen %s/%s: %.1f s Rechenzeit für %.1f s Audio (RTF %.2f, %d Abschnitt%s), "
            "davon Klang-Decoder %.1f s",
            repo_id.split("-12Hz-")[-1], label, took, dur, took / dur if dur else 0.0,
            len(chunks), "" if len(chunks) == 1 else "e", self._decode_s,
        )
        return samples


WARMUP_TEXTS = [
    "Hallo.",
    "Das ist ein kurzer Satz zum Aufwärmen.",
    "Und hier kommt ein etwas längerer Satz, damit auch größere Längen schon "
    "einmal durchgerechnet sind, bevor die erste echte Nachricht kommt.",
]


class TTS:
    def __init__(self) -> None:
        # Pipelines pro (Backbone, Sprache) – Laden kostet Zeit/RAM,
        # deshalb einmalig und gecacht.
        self._pipelines: dict[tuple[str, str], object] = {}
        self._packs: dict[str, object] = {}
        self._pack_mtime: dict[str, float | None] = {}
        self._lock = threading.Lock()
        self._ready = False
        self._qwen = QwenEngine()
        from .pocket import PocketEngine

        self._pocket = PocketEngine()
        # Mehrere Rechenprozesse je Engine (CPU), siehe pool_for. Anzahl kommt
        # aus der Konfiguration (set_workers), Startwert für Qwen QWEN_WORKERS.
        self._pools: dict[str, object] = {}
        from .qwen_pool import QWEN_WORKERS

        self._workers: dict[str, int] = {"qwen": QWEN_WORKERS, "pocket": 1}
        self._pool_lock = threading.Lock()
        from collections import OrderedDict

        self._qwen_cache: OrderedDict = OrderedDict()
        self._disk = DiskCache(DISK_CACHE_DIR, DISK_CACHE_MAX_MB)
        # Gruppenberechnung (Batching): fertige Ergebnisse warten hier auf
        # ihre Nachricht; laufende Gruppen melden sich über ein Event.
        self._prefetched: OrderedDict = OrderedDict()
        self._inflight: dict[tuple, threading.Event] = {}
        self._batch_lock = threading.Lock()

    def is_ready(self) -> bool:
        return self._ready

    @staticmethod
    def _qwen_key(voice_key: str, spec: dict, temperature, instruct, text: str) -> tuple:
        # Änderungszeit der Referenz im Schlüssel: neu erzeugte Stimme oder
        # korrigierter Text -> alter Eintrag gilt nicht mehr.
        ref_stamp = tuple(
            os.path.getmtime(spec[k]) for k in ("ref_wav", "ref_txt") if k in spec
        )
        if spec.get("engine") != "qwen":
            instruct = None  # nur Preset-Stimmen verstehen Anweisungen
        if voices_mod.is_pocket(voice_key):
            from .pocket import POCKET_PARAMS

            # Qwen-Temperatur gilt hier nicht – dafür Pockets eigene Einstellungen

            # "v2": Ergebnisse aus der fehlerhaften Fassung (Kauderwelsch) verwerfen
            temperature = ("pocket-v2",) + POCKET_PARAMS
        elif voices_mod.is_ext(voice_key):
            temperature = None  # nutzen weder Temperatur noch Anweisung
        return (voice_key, ref_stamp, QWEN_SEED, QWEN_REF_DECODE_S, temperature, instruct, text)

    def _cached_path(self, voice_key: str) -> bool:
        """Läuft diese Stimme über Zwischenspeicher/Vorab-Berechnung? Qwen
        immer, Pocket nur mit mehreren Rechenprozessen."""
        return (voices_mod.is_qwen(voice_key) or voices_mod.is_ext(voice_key)
                or (voices_mod.is_pocket(voice_key) and self.parallel(voice_key)))

    def has_result(self, text: str, voice_key: str, temperature=None, instruct=None) -> bool:
        """Liegt für diesen Text schon fertiges Audio bereit (Gruppe oder Cache)?"""
        spec = voices_mod.VOICES.get(voice_key)
        if spec is None or not self._cached_path(voice_key):
            return False
        try:
            key = self._qwen_key(voice_key, spec, temperature, instruct, text)
        except OSError:
            return False
        with self._batch_lock:
            if key in self._prefetched or key in self._qwen_cache:
                return True
        return self._disk.has(key)

    def _pocket_compute(self, voice_key: str, spec: dict, text: str) -> SynthesisResult:
        """Pocket rechnen – im Hauptprozess oder in einem freien Rechenprozess."""
        pool = self.pool_for("pocket")
        t_load = time.perf_counter()
        if pool is None:
            self.load(voice_key)
        t0 = time.perf_counter()
        if pool is not None:
            samples, compute_s, load_s, _dec, sr = pool.submit(dict(
                text=text, language=spec["language"], voice=spec.get("voice"),
                ref_wav=spec.get("ref_wav")))
        else:
            samples, sr = self._pocket.synthesize(text, spec["language"], spec.get("voice"),
                                                  spec.get("ref_wav"))
            compute_s, load_s = time.perf_counter() - t0, t0 - t_load
        samples = _resample(samples, sr, SAMPLE_RATE)
        if len(samples) == 0:
            raise ValueError("Kein Audio erzeugt.")
        return SynthesisResult(
            samples=samples, duration_seconds=len(samples) / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE, compute_seconds=compute_s, load_seconds=load_s,
        )

    def _ext_compute(self, voice_key: str, spec: dict, text: str) -> SynthesisResult:
        """Engines mit eigener venv: immer in eigenen Rechenprozessen."""
        pool = self.pool_for(spec["engine"])
        samples, compute_s, load_s, _dec, sr = pool.submit(dict(
            text=text, ref_wav=spec.get("ref_wav"), ref_txt=spec.get("ref_txt"),
            ref_codes=spec.get("ref_codes")))
        samples = _resample(samples, sr, SAMPLE_RATE)
        if len(samples) == 0:
            raise ValueError("Kein Audio erzeugt.")
        return SynthesisResult(
            samples=samples, duration_seconds=len(samples) / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE, compute_seconds=compute_s, load_seconds=load_s,
        )

    def _qwen_compute(self, voice_key: str, spec: dict, text: str,
                      temperature=None, instruct=None) -> SynthesisResult:
        """Qwen (oder Pocket mit Rechenprozessen) rechnen – im Hauptprozess
        oder in einem freien Rechenprozess."""
        if voices_mod.is_pocket(voice_key):
            return self._pocket_compute(voice_key, spec, text)
        if voices_mod.is_ext(voice_key):
            return self._ext_compute(voice_key, spec, text)
        instruct = instruct if spec.get("engine") == "qwen" else None
        t_load = time.perf_counter()
        self.load(voice_key)
        t0 = time.perf_counter()
        if self.pool_active:
            samples, compute_s, load_s, decode_s = self._pools["qwen"].submit(dict(
                repo=spec["model"], speaker=spec.get("speaker"), ref_wav=spec.get("ref_wav"),
                ref_txt=spec.get("ref_txt"), temperature=temperature, instruct=instruct,
                text=text,
            ))
        else:
            samples = self._qwen.synthesize(
                text, spec["model"], spec.get("speaker"),
                spec.get("ref_wav"), spec.get("ref_txt"), temperature, instruct,
            )
            compute_s, load_s, decode_s = (time.perf_counter() - t0, t0 - t_load,
                                           self._qwen.last_decode_s)
        if len(samples) == 0:
            raise ValueError("Kein Audio erzeugt.")
        return SynthesisResult(
            samples=samples, duration_seconds=len(samples) / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE, compute_seconds=compute_s, load_seconds=load_s,
            decode_seconds=decode_s,
        )

    def prefetch_one(self, text: str, voice_key: str,
                     temperature: float | None = None, instruct: str | None = None) -> bool:
        """Text vorab berechnen (blockierend, im Thread aufrufen) – für die
        parallelen Rechenprozesse. synthesize() holt das Ergebnis später ab
        oder wartet, bis es fertig ist."""
        if voice_key not in voices_mod.VOICES or not self._cached_path(voice_key):
            return False
        spec = voices_mod.VOICES[voice_key]
        key = self._qwen_key(voice_key, spec, temperature, instruct, text)
        with self._batch_lock:
            if (key in self._qwen_cache or key in self._prefetched or key in self._inflight
                    or self._disk.has(key)):
                return False
            event = self._inflight[key] = threading.Event()
        try:
            result = self._qwen_compute(voice_key, spec, text, temperature, instruct)
            self._disk.put(key, result.samples)
            with self._batch_lock:
                self._prefetched[key] = result
                while len(self._prefetched) > QWEN_CACHE_SIZE:
                    self._prefetched.popitem(last=False)
            return True
        except Exception as exc:  # die Nachricht rechnet dann beim Abspielen einzeln
            log.warning("Vorab-Berechnung fehlgeschlagen: %s", exc)
            return False
        finally:
            with self._batch_lock:
                self._inflight.pop(key, None)
            event.set()

    def _remember_locked(self, key: tuple, samples) -> None:
        """Ergebnis im RAM-Zwischenspeicher ablegen (Aufrufer hält _batch_lock)."""
        self._qwen_cache[key] = samples
        self._qwen_cache.move_to_end(key)
        while len(self._qwen_cache) > QWEN_CACHE_SIZE:
            self._qwen_cache.popitem(last=False)

    def forget_clone(self, name: str) -> None:
        """Nach dem Löschen einer eigenen Stimme: vorberechnete Referenz und
        zwischengespeicherte Sätze aus dem RAM werfen."""
        keys = voices_mod.clone_keys(name)
        with self._batch_lock:
            for store in (self._qwen_cache, self._prefetched):
                for key in [k for k in list(store) if k[0] in keys]:
                    store.pop(key, None)
        self._pocket.forget(str(voices_mod.ref_dir() / f"{name}.wav"))
        prompts = self._qwen._clone_prompts
        for key in [k for k in prompts if Path(k[1]).stem == name]:
            prompts.pop(key, None)

    @property
    def qwen_device(self) -> str | None:
        if self.pool_active:
            return f"CPU · {self._pools['qwen'].workers} Prozesse"
        return self._qwen.device_name

    # ------------------------------------------------ Rechenprozesse je Engine
    POOL_ENGINES = {"qwen": "Qwen", "pocket": "Pocket TTS"}

    @staticmethod
    def _engine_of(voice_key: str) -> str | None:
        if voices_mod.is_qwen(voice_key):
            return "qwen"
        if voices_mod.is_pocket(voice_key):
            return "pocket"
        if voices_mod.is_ext(voice_key):
            return voices_mod.engine(voice_key)
        return None

    def _qwen_on_gpu(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available() and os.environ.get("QWEN_DEVICE", "auto") != "cpu"
        except Exception:
            return False

    def pool_for(self, engine: str):
        """Rechenprozesse dieser Engine, oder None (alles im Hauptprozess)."""
        with self._pool_lock:
            pool = self._pools.get(engine)
            if pool is not None:
                return pool
            n = self._workers.get(engine, 1)
            if engine in voices_mod.EXT_ENGINES:  # laufen immer in eigenen Prozessen
                from .ext_pool import ExtPool

                pool = self._pools[engine] = ExtPool(engine, n, voices_mod.EXT_PYTHON[engine])
                return pool
            if n <= 1:
                return None
            from .qwen_pool import QwenPool

            if engine == "qwen":
                if self._qwen_on_gpu():
                    return None  # GPU: ein Prozess ist schneller
                try:
                    import torch

                    threads = int(os.environ.get("QWEN_THREADS", "0") or 0) or torch.get_num_threads()
                except Exception:
                    return None
            else:
                from .pocket import POCKET_THREADS

                threads = POCKET_THREADS
            pool = self._pools[engine] = QwenPool(n, threads, engine=engine)
            return pool

    def set_workers(self, engine: str, count: int, restart: bool = True) -> None:
        """Anzahl der Rechenprozesse setzen; laufende Prozesse werden beendet
        und beim nächsten Auftrag mit der neuen Anzahl gestartet."""
        if engine not in self.POOL_ENGINES:
            raise ValueError(f"Unbekannte Engine: {engine}")
        count = min(max(int(count), 1), 6)
        with self._pool_lock:
            self._workers[engine] = count
            old = self._pools.pop(engine, None) if restart else None
        if old is not None:
            old.shutdown()

    def workers_state(self) -> dict:
        out = {}
        for engine, label in self.POOL_ENGINES.items():
            pool = self._pools.get(engine)
            note = ""
            if engine in voices_mod.EXT_ENGINES and not Path(voices_mod.EXT_PYTHON[engine]).is_file():
                continue  # nicht im Image installiert
            if engine == "qwen" and self._qwen_on_gpu():
                note = "GPU erkannt – Qwen rechnet dort immer in einem Prozess."
            out[engine] = {"label": label, "count": self._workers.get(engine, 1),
                           "running": pool.workers if pool is not None and pool._started else 0,
                           "note": note}
        return out

    def parallel(self, voice_key: str) -> bool:
        """Rechnet diese Stimme mit mehreren Prozessen?"""
        engine = self._engine_of(voice_key) if voice_key in voices_mod.VOICES else None
        return (engine is not None and self._workers.get(engine, 1) > 1
                and self.pool_for(engine) is not None)

    def workers_for(self, voice_key: str) -> int:
        engine = self._engine_of(voice_key) if voice_key in voices_mod.VOICES else None
        pool = self.pool_for(engine) if engine else None
        return pool.workers if pool is not None else 1

    @property
    def pool_active(self) -> bool:
        """Mehrere Qwen-Rechenprozesse aktiv?"""
        return self.pool_for("qwen") is not None

    @property
    def _pool(self):
        return self._pools.get("qwen")

    @property
    def pool_workers(self) -> int:
        return self._pool.workers if self.pool_active else 1

    warming = False

    def warmup(self, voice_key: str) -> None:
        """Nach dem Start: Qwen-Modell laden und auf der GPU mit ein paar
        Probesätzen verschiedener Länge aufwärmen (torch.compile kompiliert
        dabei), damit der erste echte Satz nicht warten muss."""
        # Reihenfolge wichtig: is_available liest eigene Stimmen erst ein –
        # vorher kennt is_qwen "klon-…"-Schlüssel noch nicht (KeyError, das
        # Aufwärmen fiel dadurch stillschweigend aus).
        try:
            if not voices_mod.is_available(voice_key) or not voices_mod.is_qwen(voice_key):
                log.info("Kein Aufwärmen: gewählte Stimme %s ist keine Qwen-Stimme", voice_key)
                return
        except Exception as exc:
            log.warning("Aufwärmen übersprungen: %s", exc)
            return
        self.warming = True
        t0 = time.perf_counter()
        log.info("Aufwärmen: lade %s …", voice_key)
        try:
            if self.pool_active:
                spec = voices_mod.VOICES[voice_key]
                from concurrent.futures import ThreadPoolExecutor

                args = dict(repo=spec["model"], speaker=spec.get("speaker"),
                            ref_wav=spec.get("ref_wav"), ref_txt=spec.get("ref_txt"),
                            text=WARMUP_TEXTS[0])
                allowed = sorted(self._pool._allowed(spec["model"]))
                # Gleichzeitig laden: beim Laden braucht ein Modell kurzzeitig
                # deutlich mehr Speicher. 0.6B ist klein -> alle auf einmal;
                # 1.7B höchstens 3 gleichzeitig (QWEN_LOAD_PARALLEL überschreibt)
                par = int(os.environ.get("QWEN_LOAD_PARALLEL", "0") or 0) or (
                    len(allowed) if "0.6B" in spec["model"] else 3)
                with ThreadPoolExecutor(max(1, min(par, len(allowed)))) as ex:
                    list(ex.map(lambda i: self._pool.submit(dict(args), worker=i), allowed))
                log.info("Qwen in %d Rechenprozessen geladen in %.0f s",
                         len(allowed), time.perf_counter() - t0)
                return
            self.load(voice_key)
            if (self._qwen.device_name or "CPU") == "CPU":
                log.info("Qwen-Modell vorgeladen (CPU, kein Aufwärmen nötig)")
                return
            log.info("Aufwärmen der GPU mit Probesätzen …")
            spec = voices_mod.VOICES[voice_key]
            for text in WARMUP_TEXTS:
                self._qwen.synthesize(text, spec["model"], spec.get("speaker"),
                                      spec.get("ref_wav"), spec.get("ref_txt"))
            log.info("Aufwärmen fertig in %.0f s", time.perf_counter() - t0)
        except Exception as exc:  # Aufwärmen darf den Betrieb nie verhindern
            log.warning("Aufwärmen fehlgeschlagen: %s", exc)
        finally:
            self.warming = False

    def load(self, voice_key: str) -> None:
        """Pipeline + Voicepack für eine Stimme laden (blockierend)."""
        spec = voices_mod.VOICES[voice_key]
        if voices_mod.is_ext(voice_key):
            if not voices_mod.is_available(voice_key):
                raise FileNotFoundError(
                    f"Für die Stimme '{spec['label']}' fehlt die Engine im Image oder die Referenz.")
            self.pool_for(spec["engine"]).start()  # Prozesse laden ihr Modell sofort
            self._ready = True
            return
        if voices_mod.is_pocket(voice_key):
            if not voices_mod.is_available(voice_key):
                raise FileNotFoundError("Pocket TTS ist nicht installiert (Paket pocket-tts).")
            if self.pool_for("pocket") is None:  # sonst laden die Rechenprozesse selbst
                self._pocket.load(spec["language"])
            self._ready = True
            return
        if voices_mod.is_qwen(voice_key):
            if not voices_mod.is_available(voice_key):
                raise FileNotFoundError(
                    f"Für die Stimme '{spec['label']}' fehlt das Qwen-Modell oder das Paket qwen-tts."
                )
            if self.pool_active:
                self._pool.start()  # Prozesse laden ihr Modell beim ersten Auftrag
            else:
                self._qwen.load(spec["model"])
            self._ready = True
            return

        cache_key = (spec["backbone"], spec["lang"])

        with self._lock:
            if cache_key not in self._pipelines:
                if not voices_mod.is_available(voice_key):
                    raise FileNotFoundError(
                        f"Für die Stimme '{spec['label']}' fehlen Modelldateien. "
                        "Bitte install.sh erneut ausführen."
                    )

                from kokoro import KModel, KPipeline

                weights = voices_mod.backbone_path(voice_key)
                log.info("Lade Stimm-Modell: %s (%s)", spec["label"], weights.name)
                kmodel = KModel(
                    repo_id=REPO_ID,
                    config=str(voices_mod.config_path()),
                    model=str(weights),
                ).eval()
                self._pipelines[cache_key] = KPipeline(
                    lang_code=spec["lang"], repo_id=REPO_ID, model=kmodel
                )

            if voice_key not in self._packs:
                import torch

                self._packs[voice_key] = torch.load(
                    voices_mod.voicepack_path(voice_key),
                    map_location="cpu",
                    weights_only=True,
                )

        self._ready = True

    def synthesize(self, text: str, voice_key: str, speed: float = 1.0,
                   temperature: float | None = None,
                   instruct: str | None = None) -> SynthesisResult:
        if voice_key.startswith("klon-"):
            voices_mod.sync_clone_voices()
        if voice_key not in voices_mod.VOICES:
            voice_key = voices_mod.DEFAULT_VOICE

        spec = voices_mod.VOICES[voice_key]
        if voices_mod.is_pocket(voice_key) and not self.parallel(voice_key):
            t_load = time.perf_counter()
            self.load(voice_key)
            t0 = time.perf_counter()
            samples, sr = self._pocket.synthesize(text, spec["language"], spec.get("voice"),
                                                  spec.get("ref_wav"))
            samples = _resample(samples, sr, SAMPLE_RATE)
            if len(samples) == 0:
                raise ValueError("Kein Audio erzeugt.")
            return SynthesisResult(
                samples=samples,
                duration_seconds=len(samples) / SAMPLE_RATE,
                sample_rate=SAMPLE_RATE,
                compute_seconds=time.perf_counter() - t0,
                load_seconds=t0 - t_load,
            )
        if self._cached_path(voice_key):
            # Qwen kennt keine Sprechgeschwindigkeit – der Regler wirkt hier nicht.
            # Cache-Schlüssel enthält bei eigenen Stimmen die Änderungszeit der
            # Referenz – wird sie neu erzeugt oder der Text korrigiert, gilt der
            # alte Eintrag nicht mehr.
            key = self._qwen_key(voice_key, spec, temperature, instruct, text)
            # Läuft für diesen Text gerade eine Berechnung (Vorab oder ein
            # anderer Abschnitt mit gleichem Text)? Dann darauf warten statt
            # doppelt zu rechnen. Alles unter _batch_lock – synthesize läuft in
            # mehreren Threads gleichzeitig.
            while True:
                with self._batch_lock:
                    event = self._inflight.get(key)
                    if event is None:
                        ready = self._prefetched.pop(key, None)
                        if ready is not None:
                            self._remember_locked(key, ready.samples)
                            return ready
                        samples = self._qwen_cache.get(key)
                        if samples is not None:
                            self._qwen_cache.move_to_end(key)
                            return SynthesisResult(
                                samples=samples, duration_seconds=len(samples) / SAMPLE_RATE,
                                sample_rate=SAMPLE_RATE, cached=True,
                            )
                        mine = self._inflight[key] = threading.Event()
                        break
                event.wait()
            try:
                samples = self._disk.get(key)
                if samples is not None:
                    with self._batch_lock:
                        self._remember_locked(key, samples)
                    return SynthesisResult(
                        samples=samples, duration_seconds=len(samples) / SAMPLE_RATE,
                        sample_rate=SAMPLE_RATE, cached=True,
                    )
                result = self._qwen_compute(voice_key, spec, text, temperature, instruct)
                self._disk.put(key, result.samples)
                with self._batch_lock:
                    self._remember_locked(key, result.samples)
                return result
            finally:
                with self._batch_lock:
                    self._inflight.pop(key, None)
                mine.set()

        t_load = time.perf_counter()
        cache_key = (spec["backbone"], spec["lang"])
        if spec.get("voicepack_file"):
            # eigenes Stimmpaket: wird während der Suche ersetzt -> neu laden
            try:
                mtime = os.path.getmtime(spec["voicepack_file"])
            except OSError:
                mtime = None
            if self._pack_mtime.get(voice_key) != mtime:
                self._packs.pop(voice_key, None)
                self._pack_mtime[voice_key] = mtime
        if cache_key not in self._pipelines or voice_key not in self._packs:
            self.load(voice_key)
        t0 = time.perf_counter()

        import numpy as np

        pipeline = self._pipelines[cache_key]
        pack = self._packs[voice_key]

        chunks = []
        for _graphemes, _phonemes, audio in pipeline(text, voice=pack, speed=float(speed)):
            chunks.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else audio)

        if not chunks:
            raise ValueError("Kein Audio erzeugt (leerer Text nach Phonemisierung?).")

        samples = np.concatenate(chunks).astype(np.float32)
        return SynthesisResult(
            samples=samples,
            duration_seconds=len(samples) / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE,
            compute_seconds=time.perf_counter() - t0,
            load_seconds=t0 - t_load,
        )
