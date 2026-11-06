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
                 "load_seconds")

    def __init__(self, samples, duration_seconds: float, sample_rate: int,
                 compute_seconds: float = 0.0, load_seconds: float = 0.0):
        self.samples = samples
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.compute_seconds = compute_seconds  # reine Sprachberechnung
        self.load_seconds = load_seconds        # Modell laden (nur beim ersten Mal)


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


def _split_for_qwen(text: str, max_chars: int = QWEN_CHUNK_CHARS) -> list[str]:
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


class QwenEngine:
    """Qwen3-TTS (CustomVoice) auf CPU. Hält nur ein Modell im RAM – beim
    Wechsel zwischen 0.6B und 1.7B wird das vorherige freigegeben.

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
            gc.collect()
            threads = int(os.environ.get("QWEN_THREADS", "0") or 0)
            if threads > 0:
                torch.set_num_threads(threads)
            log.info("Lade Qwen3-TTS: %s (%d Threads)", repo_id, torch.get_num_threads())
            model = Qwen3TTSModel.from_pretrained(
                str(snapshot), device_map="cpu", dtype=torch.float32
            )
            self._model = model
            self._repo = repo_id

    def synthesize(self, text: str, repo_id: str, speaker: str):
        import numpy as np

        self.load(repo_id)
        chunks = _split_for_qwen(text)
        parts = []
        t0 = time.perf_counter()
        for i, chunk in enumerate(chunks, 1):
            with self._lock:
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
        log.info(
            "Qwen %s/%s: %.1f s Rechenzeit für %.1f s Audio (RTF %.2f, %d Abschnitt%s)",
            repo_id.split("-12Hz-")[-1], speaker, took, dur, took / dur if dur else 0.0,
            len(chunks), "" if len(chunks) == 1 else "e",
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

    def is_ready(self) -> bool:
        return self._ready

    def load(self, voice_key: str) -> None:
        """Pipeline + Voicepack für eine Stimme laden (blockierend)."""
        spec = voices_mod.VOICES[voice_key]
        if voices_mod.engine(voice_key) == "qwen":
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
        if voice_key not in voices_mod.VOICES:
            voice_key = voices_mod.DEFAULT_VOICE

        spec = voices_mod.VOICES[voice_key]
        if voices_mod.engine(voice_key) == "qwen":
            # Qwen kennt keine Sprechgeschwindigkeit – der Regler wirkt hier nicht.
            t_load = time.perf_counter()
            self.load(voice_key)
            t0 = time.perf_counter()
            samples = self._qwen.synthesize(text, spec["model"], spec["speaker"])
            if len(samples) == 0:
                raise ValueError("Kein Audio erzeugt.")
            return SynthesisResult(
                samples=samples,
                duration_seconds=len(samples) / SAMPLE_RATE,
                sample_rate=SAMPLE_RATE,
                compute_seconds=time.perf_counter() - t0,
                load_seconds=t0 - t_load,
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
