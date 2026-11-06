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
    """Text an Satzenden (notfalls Kommas/Leerzeichen) in Abschnitte teilen.
    first_max: kürzere Obergrenze nur für den ersten Abschnitt."""
    if first_max and len(text.strip()) > first_max:
        # Leerraum vereinheitlichen, damit der Rest exakt hinter dem Kopf beginnt
        rest = re.sub(r"\s+", " ", text).strip()
        chunks: list[str] = []
        limit = first_max
        while rest:
            if len(rest) <= limit:
                chunks.append(rest)
                break
            head, rest = _take_head(rest, limit)
            chunks.append(head)
            # vom tatsächlich erzeugten Abschnitt aus wachsen, aber nie unter die
            # bisherige Grenze schrumpfen (sonst zerfällt ein Text mit langen
            # Sätzen in immer kleinere Stücke mitten im Satz)
            limit = min(max_chars, max(int(limit * 1.2), int(len(head) * QWEN_CHUNK_GROWTH)))
        # winzigen Rest an den vorletzten Abschnitt hängen, wenn er passt
        if len(chunks) > 1 and len(chunks[-1]) < 40 and len(chunks[-2]) + 1 + len(chunks[-1]) <= max_chars:
            tail = chunks.pop()  # erst entnehmen: chunks[-2] += chunks.pop() trifft
            chunks[-1] += " " + tail  # sonst nach dem pop den falschen Index
        return chunks
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = [s for s in re.split(r"(?<=[.!?…;:])\s+|\n+", text) if s.strip()]
    pieces: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Überlange Einzelsätze an Kommas, sonst an Leerzeichen zerlegen
        while len(sentence) > max_chars:
            cut = sentence.rfind(", ", 0, max_chars)
            if cut < max_chars // 3:
                cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            pieces.append(sentence[:cut + 1].strip())
            sentence = sentence[cut + 1:].strip()
        if sentence:
            pieces.append(sentence)
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 1 + len(piece) <= max_chars:
            chunks[-1] += " " + piece
        else:
            chunks.append(piece)
    return chunks


def _take_head(text: str, limit: int) -> tuple[str, str]:
    """Ersten Satz (oder Satzteil bis Komma/Leerzeichen) bis limit Zeichen."""
    sentences = [s for s in re.split(r"(?<=[.!?…;:])\s+|\n+", text) if s.strip()]
    head = sentences[0].strip()
    # mehrere kurze Sätze dürfen zusammen in den ersten Abschnitt
    for s in sentences[1:]:
        # sehr kurzen Anfang ("Bild:") mit dem nächsten Satz verbinden und
        # notfalls unten am Komma/Leerzeichen kürzen
        if len(head) + 1 + len(s.strip()) > limit and len(head) >= limit // 3:
            break
        head += " " + s.strip()
        if len(head) > limit:
            break
    if len(head) > limit:
        cut = head.rfind(", ", 0, limit)
        if cut < limit // 3:
            cut = head.rfind(" ", 0, limit)
        if cut > 0:
            head = head[:cut + 1].strip()
    assert text.startswith(head)
    return head, text[len(head):].strip()


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
    if chunk[-1] not in ".!?…":
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


    def synthesize_batch(self, texts: list[str], repo_id: str, speaker: str | None = None,
                         ref_wav: str | None = None, ref_txt: str | None = None,
                         temperature: float | None = None, instruct: str | None = None):
        """Mehrere kurze Texte in EINEM Durchlauf erzeugen. Auf der CPU kostet
        ein Schritt für 8 Texte kaum mehr als für einen (Verwaltungsaufwand
        statt Rechenleistung bremst) – der Durchsatz steigt entsprechend.
        Liefert [(samples, Stimm-Ähnlichkeit oder None)], Rechenzeit."""
        import numpy as np
        import torch

        self.load(repo_id)
        n = len(texts)
        budget = max(_qwen_token_budget(t) for t in texts)
        with self._lock:
            trim = 0
            prompt = None
            if ref_wav:
                prompt = self._clone_prompt(ref_wav, ref_txt)
                ref_code = getattr(prompt[0], "ref_code", None) if prompt else None
                if QWEN_REF_DECODE_S > 0 and ref_code is not None:
                    trim = max(0, int(ref_code.shape[0])
                               - int(QWEN_REF_DECODE_S * QWEN_STEPS_PER_SECOND))
                gen = lambda: self._model.generate_voice_clone(  # noqa: E731
                    text=[_qwen_text(t) for t in texts], language=["German"] * n,
                    voice_clone_prompt=list(prompt) * n,
                    max_new_tokens=budget, temperature=temperature,
                )
            else:
                kwargs = {"instruct": [instruct] * n} if instruct else {}
                gen = lambda: self._model.generate_custom_voice(  # noqa: E731
                    text=[_qwen_text(t) for t in texts], language=["German"] * n,
                    speaker=[speaker] * n, max_new_tokens=budget,
                    temperature=temperature, **kwargs,
                )
            self._decode_s = 0.0
            t0 = time.perf_counter()
            self._trim_codes = trim
            try:
                torch.manual_seed(QWEN_SEED)
                try:
                    wavs, sr = gen()
                except Exception as exc:
                    if not self._compiled:
                        raise
                    self._uncompile(exc)
                    torch.manual_seed(QWEN_SEED)
                    wavs, sr = gen()
            finally:
                self._trim_codes = 0
            took = time.perf_counter() - t0
            out = []
            for wav in wavs:
                wav = np.asarray(wav, dtype=np.float32)
                if trim:  # Auffüll-Stille der Referenz entfernen
                    wav = np.trim_zeros(wav, "f")
                sim = (self._speaker_similarity(prompt[0], wav, int(sr))
                       if prompt and QWEN_SPK_MIN > 0 else None)
                audio = _resample(wav, int(sr), SAMPLE_RATE)
                audio = np.concatenate([audio, np.zeros(int(0.15 * SAMPLE_RATE), dtype=np.float32)])
                out.append((audio, sim))
        dur = sum(len(a) for a, _ in out) / SAMPLE_RATE
        log.info("Qwen %s Gruppe: %d Nachrichten, %.1f s Rechenzeit für %.1f s Audio (RTF %.2f)",
                 repo_id.split("-12Hz-")[-1], n, took, dur, took / dur if dur else 0.0)
        return out, took


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
        self._lock = threading.Lock()
        self._ready = False
        self._qwen = QwenEngine()
        from .pocket import PocketEngine

        self._pocket = PocketEngine()
        from collections import OrderedDict

        self._qwen_cache: OrderedDict = OrderedDict()
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
        return (voice_key, ref_stamp, QWEN_SEED, QWEN_REF_DECODE_S, temperature, instruct, text)

    def prefetch_batch(self, texts: list[str], voice_key: str,
                       temperature: float | None = None, instruct: str | None = None) -> int:
        """Mehrere Texte gemeinsam vorab berechnen (blockierend, im Thread
        aufrufen). synthesize() holt die Ergebnisse später ab oder wartet auf
        die laufende Gruppe. Liefert die Zahl tatsächlich berechneter Texte."""
        if not voices_mod.is_qwen(voice_key) or voice_key not in voices_mod.VOICES:
            return 0
        spec = voices_mod.VOICES[voice_key]
        with self._batch_lock:
            todo: dict[tuple, str] = {}
            for text in texts:
                key = self._qwen_key(voice_key, spec, temperature, instruct, text)
                if (key in self._qwen_cache or key in self._prefetched
                        or key in self._inflight or key in todo):
                    continue
                todo[key] = text
            for key in todo:
                self._inflight[key] = threading.Event()
        if not todo:
            return 0
        try:
            self.load(voice_key)
            results, took = self._qwen.synthesize_batch(
                list(todo.values()), spec["model"], spec.get("speaker"),
                spec.get("ref_wav"), spec.get("ref_txt"), temperature,
                instruct if spec.get("engine") == "qwen" else None,
            )
            total = sum(len(a) for a, _ in results) or 1
            with self._batch_lock:
                for key, (audio, sim) in zip(todo, results):
                    # Klingt ein Ergebnis nicht nach der Vorlage, wird es nicht
                    # übernommen – die Nachricht rechnet dann einzeln (mit
                    # erneutem Versuch, siehe Stimm-Prüfung).
                    if sim is not None and sim < QWEN_SPK_MIN:
                        log.warning("Gruppe: Stimm-Ähnlichkeit %.2f zu niedrig – Nachricht "
                                    "wird einzeln neu erzeugt", sim)
                        continue
                    if len(audio) == 0:
                        continue
                    self._prefetched[key] = SynthesisResult(
                        samples=audio, duration_seconds=len(audio) / SAMPLE_RATE,
                        sample_rate=SAMPLE_RATE,
                        compute_seconds=took * len(audio) / total,
                        decode_seconds=self._qwen.last_decode_s * len(audio) / total,
                        batch=len(todo),
                    )
                while len(self._prefetched) > QWEN_CACHE_SIZE:
                    self._prefetched.popitem(last=False)
            return len(todo)
        except Exception as exc:  # Gruppe gescheitert -> Nachrichten rechnen einzeln
            log.warning("Gruppenberechnung fehlgeschlagen: %s", exc)
            return 0
        finally:
            with self._batch_lock:
                for key in todo:
                    event = self._inflight.pop(key, None)
                    if event:
                        event.set()

    def forget_clone(self, name: str) -> None:
        """Nach dem Löschen einer eigenen Stimme: vorberechnete Referenz und
        zwischengespeicherte Sätze aus dem RAM werfen."""
        keys = voices_mod.clone_keys(name)
        for key in [k for k in list(self._qwen_cache) if k[0] in keys]:
            self._qwen_cache.pop(key, None)
        self._pocket.forget(str(voices_mod.ref_dir() / f"{name}.wav"))
        prompts = self._qwen._clone_prompts
        for key in [k for k in prompts if Path(k[1]).stem == name]:
            prompts.pop(key, None)

    @property
    def qwen_device(self) -> str | None:
        return self._qwen.device_name

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
        if voices_mod.is_pocket(voice_key):
            if not voices_mod.is_available(voice_key):
                raise FileNotFoundError("Pocket TTS ist nicht installiert (Paket pocket-tts).")
            self._pocket.load(spec["language"])
            self._ready = True
            return
        if voices_mod.is_qwen(voice_key):
            if not voices_mod.is_available(voice_key):
                raise FileNotFoundError(
                    f"Für die Stimme '{spec['label']}' fehlt das Qwen-Modell oder das Paket qwen-tts."
                )
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
        if voices_mod.is_pocket(voice_key):
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
        if voices_mod.is_qwen(voice_key):
            # Qwen kennt keine Sprechgeschwindigkeit – der Regler wirkt hier nicht.
            # Cache-Schlüssel enthält bei eigenen Stimmen die Änderungszeit der
            # Referenz – wird sie neu erzeugt oder der Text korrigiert, gilt der
            # alte Eintrag nicht mehr.
            key = self._qwen_key(voice_key, spec, temperature, instruct, text)
            # Läuft für diesen Text gerade eine Gruppenberechnung? Dann darauf
            # warten statt doppelt zu rechnen – und das Ergebnis abholen.
            with self._batch_lock:
                event = self._inflight.get(key)
            if event is not None:
                event.wait()
            with self._batch_lock:
                ready = self._prefetched.pop(key, None)
            if ready is not None:
                self._qwen_cache[key] = ready.samples
                while len(self._qwen_cache) > QWEN_CACHE_SIZE:
                    self._qwen_cache.popitem(last=False)
                return ready
            if key in self._qwen_cache:
                self._qwen_cache.move_to_end(key)
                samples = self._qwen_cache[key]
                return SynthesisResult(
                    samples=samples, duration_seconds=len(samples) / SAMPLE_RATE,
                    sample_rate=SAMPLE_RATE, cached=True,
                )
            t_load = time.perf_counter()
            self.load(voice_key)
            t0 = time.perf_counter()
            samples = self._qwen.synthesize(
                text, spec["model"], spec.get("speaker"),
                spec.get("ref_wav"), spec.get("ref_txt"), temperature,
                instruct if spec.get("engine") == "qwen" else None,
            )
            if len(samples) == 0:
                raise ValueError("Kein Audio erzeugt.")
            self._qwen_cache[key] = samples
            while len(self._qwen_cache) > QWEN_CACHE_SIZE:
                self._qwen_cache.popitem(last=False)
            return SynthesisResult(
                samples=samples,
                duration_seconds=len(samples) / SAMPLE_RATE,
                sample_rate=SAMPLE_RATE,
                compute_seconds=time.perf_counter() - t0,
                load_seconds=t0 - t_load,
                decode_seconds=self._qwen.last_decode_s,
            )

        t_load = time.perf_counter()
        cache_key = (spec["backbone"], spec["lang"])
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
