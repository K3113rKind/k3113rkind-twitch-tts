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
                 "load_seconds", "cached", "decode_seconds")

    def __init__(self, samples, duration_seconds: float, sample_rate: int,
                 compute_seconds: float = 0.0, load_seconds: float = 0.0,
                 cached: bool = False, decode_seconds: float = 0.0):
        self.samples = samples
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.compute_seconds = compute_seconds  # reine Sprachberechnung
        self.load_seconds = load_seconds        # Modell laden (nur beim ersten Mal)
        self.cached = cached                    # aus dem Zwischenspeicher, nicht neu berechnet
        self.decode_seconds = decode_seconds    # davon Klangerzeugung (Qwen-Decoder)


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


def split_for_qwen(text: str, max_chars: int = QWEN_CHUNK_CHARS) -> list[str]:
    """Text an Satzenden (notfalls Kommas/Leerzeichen) in Abschnitte teilen."""
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


# Qwen würfelt bei jeder Erzeugung (Sampling, Temperatur 0.9) – derselbe Satz
# klang deshalb jedes Mal anders betont. Mit festem Startwert vor jedem Satz
# ist das Ergebnis für denselben Text und dieselbe Stimme immer gleich;
# verschiedene Sätze klingen trotzdem natürlich unterschiedlich.
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
        # bfloat16 auf der GPU: halber Speicher, deutlich schneller; von Qwen
        # so empfohlen. Auf der CPU bleibt es bei float32 (Skylake ohne bf16).
        return "cuda:0", torch.bfloat16, torch.cuda.get_device_name(0)
    return "cpu", torch.float32, "CPU"


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
            self.device_name = device_name
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

    def _clone_prompt(self, ref_wav: str, ref_txt: str):
        """Referenz einmal vorberechnen (spart das bei jedem Satz)."""
        from pathlib import Path

        key = (self._repo, ref_wav, Path(ref_wav).stat().st_mtime, Path(ref_txt).stat().st_mtime)
        if key not in self._clone_prompts:
            text = Path(ref_txt).read_text(encoding="utf-8").strip()
            t0 = time.perf_counter()
            self._clone_prompts[key] = self._model.create_voice_clone_prompt(
                ref_audio=ref_wav, ref_text=text, x_vector_only_mode=False
            )
            log.info("Referenz %s vorbereitet in %.1f s", Path(ref_wav).name, time.perf_counter() - t0)
        return self._clone_prompts[key]

    def synthesize(self, text: str, repo_id: str, speaker: str | None = None,
                   ref_wav: str | None = None, ref_txt: str | None = None):
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
                torch.manual_seed(QWEN_SEED)
                if ref_wav:
                    prompt = self._clone_prompt(ref_wav, ref_txt)
                    ref_code = getattr(prompt[0], "ref_code", None) if prompt else None
                    keep = int(QWEN_REF_DECODE_S * QWEN_STEPS_PER_SECOND)
                    if QWEN_REF_DECODE_S > 0 and ref_code is not None:
                        self._trim_codes = max(0, int(ref_code.shape[0]) - keep)
                    try:
                        wavs, sr = self._model.generate_voice_clone(
                            text=chunk, language="German",
                            voice_clone_prompt=prompt,
                            max_new_tokens=_qwen_token_budget(chunk),
                        )
                    finally:
                        self._trim_codes = 0
                else:
                    wavs, sr = self._model.generate_custom_voice(
                        text=chunk, language="German", speaker=speaker,
                        max_new_tokens=_qwen_token_budget(chunk),
                    )
            audio = _resample(wavs[0].astype("float32"), int(sr), SAMPLE_RATE)
            dur = len(audio) / SAMPLE_RATE
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


class TTS:
    def __init__(self) -> None:
        # Pipelines pro (Backbone, Sprache) – Laden kostet Zeit/RAM,
        # deshalb einmalig und gecacht.
        self._pipelines: dict[tuple[str, str], object] = {}
        self._packs: dict[str, object] = {}
        self._lock = threading.Lock()
        self._ready = False
        self._qwen = QwenEngine()
        from collections import OrderedDict

        self._qwen_cache: OrderedDict = OrderedDict()

    def is_ready(self) -> bool:
        return self._ready

    @property
    def qwen_device(self) -> str | None:
        return self._qwen.device_name

    def load(self, voice_key: str) -> None:
        """Pipeline + Voicepack für eine Stimme laden (blockierend)."""
        spec = voices_mod.VOICES[voice_key]
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

    def synthesize(self, text: str, voice_key: str, speed: float = 1.0) -> SynthesisResult:
        if voice_key.startswith("klon-"):
            voices_mod.sync_clone_voices()
        if voice_key not in voices_mod.VOICES:
            voice_key = voices_mod.DEFAULT_VOICE

        spec = voices_mod.VOICES[voice_key]
        if voices_mod.is_qwen(voice_key):
            # Qwen kennt keine Sprechgeschwindigkeit – der Regler wirkt hier nicht.
            # Cache-Schlüssel enthält bei eigenen Stimmen die Änderungszeit der
            # Referenz – wird sie neu erzeugt oder der Text korrigiert, gilt der
            # alte Eintrag nicht mehr.
            ref_stamp = tuple(
                os.path.getmtime(spec[k]) for k in ("ref_wav", "ref_txt") if k in spec
            )
            key = (voice_key, ref_stamp, QWEN_SEED, QWEN_REF_DECODE_S, text)
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
                spec.get("ref_wav"), spec.get("ref_txt"),
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
