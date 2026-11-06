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


class QwenEngine:
    """Qwen3-TTS (CustomVoice) auf CPU. Hält nur ein Modell im RAM – beim
    Wechsel zwischen 0.6B und 1.7B wird das vorherige freigegeben.

    Stellschrauben:
      QWEN_THREADS  (Umgebungsvariable, .env) Anzahl Torch-Threads; 0 = alle
                    sichtbaren Kerne. Oft schneller: nur echte Kerne eines
                    CPU-Sockels statt Hyper-Threads über beide Sockel verteilt.
      int8          (Schalter in der GUI) Sprachmodell (Talker) dynamisch auf
                    int8 quantisieren. Experimentell: meist schneller auf CPU,
                    Klang kann leiden. Umschalten lädt das Modell neu.
    """

    def __init__(self) -> None:
        self._loaded: tuple[str, bool] | None = None  # (Modell, int8)
        self._model = None
        self._lock = threading.Lock()

    def load(self, repo_id: str, int8: bool = False) -> None:
        with self._lock:
            if self._loaded == (repo_id, int8):
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
            log.info(
                "Lade Qwen3-TTS: %s (%d Threads, int8 %s)",
                repo_id, torch.get_num_threads(), "an" if int8 else "aus",
            )
            model = Qwen3TTSModel.from_pretrained(
                str(snapshot), device_map="cpu", dtype=torch.float32
            )
            if int8:
                model.model.talker = torch.ao.quantization.quantize_dynamic(
                    model.model.talker, {torch.nn.Linear}, dtype=torch.qint8
                )
            self._model = model
            self._loaded = (repo_id, int8)

    def synthesize(self, text: str, repo_id: str, speaker: str, int8: bool = False):
        self.load(repo_id, int8)
        with self._lock:
            t0 = time.perf_counter()
            wavs, sr = self._model.generate_custom_voice(
                text=text, language="German", speaker=speaker
            )
            took = time.perf_counter() - t0
        dur = len(wavs[0]) / int(sr) if sr else 0.0
        log.info(
            "Qwen %s/%s%s: %.1f s Rechenzeit für %.1f s Audio (RTF %.2f)",
            repo_id.split("-12Hz-")[-1], speaker, " int8" if int8 else "",
            took, dur, took / dur if dur else 0.0,
        )
        return _resample(wavs[0].astype("float32"), int(sr), SAMPLE_RATE)


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

    def load(self, voice_key: str, qwen_int8: bool = False) -> None:
        """Pipeline + Voicepack für eine Stimme laden (blockierend)."""
        spec = voices_mod.VOICES[voice_key]
        if voices_mod.engine(voice_key) == "qwen":
            if not voices_mod.is_available(voice_key):
                raise FileNotFoundError(
                    f"Für die Stimme '{spec['label']}' fehlt das Qwen-Modell oder das Paket qwen-tts."
                )
            self._qwen.load(spec["model"], qwen_int8)
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
                   qwen_int8: bool = False) -> SynthesisResult:
        if voice_key not in voices_mod.VOICES:
            voice_key = voices_mod.DEFAULT_VOICE

        spec = voices_mod.VOICES[voice_key]
        if voices_mod.engine(voice_key) == "qwen":
            # Qwen kennt keine Sprechgeschwindigkeit – der Regler wirkt hier nicht.
            t_load = time.perf_counter()
            self.load(voice_key, qwen_int8)
            t0 = time.perf_counter()
            samples = self._qwen.synthesize(text, spec["model"], spec["speaker"], qwen_int8)
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
