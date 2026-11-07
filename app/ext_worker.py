"""Rechenprozess für Engines mit eigener Python-Umgebung (Chatterbox, VoxCPM2, OmniVoice).

Läuft NICHT im App-Interpreter, sondern mit dem Python der jeweiligen venv
(/opt/chatterbox) – deren Abhängigkeiten (Torch-, Transformers-
Versionen) vertragen sich nicht mit Kokoro/Qwen/Pocket. Deshalb nur
Standardbibliothek + Engine-Pakete, keine Importe aus app/.

Protokoll (eine JSON-Zeile je Auftrag auf stdin):
  {"id": 1, "text": "...", "ref_wav": "/ref/x.wav", "ref_txt": "/ref/x.txt"}
Antwort auf stdout (Bibliotheks-Ausgaben gehen nach stderr):
  @@RESULT {"id": 1, "ok": true, "npy": "/tmp/…npy", "sr": 24000,
            "compute": 1.2, "load": 0.0}
Beim Start wird das Modell sofort geladen und "@@READY" gemeldet.

Aufruf: /opt/chatterbox/bin/python app/ext_worker.py chatterbox
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback

_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr  # print() der Bibliotheken nicht ins Protokoll


def _send(prefix: str, payload: dict | None = None) -> None:
    line = prefix if payload is None else f"{prefix} {json.dumps(payload)}"
    _REAL_STDOUT.write(line + "\n")
    _REAL_STDOUT.flush()


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# -------------------------------------------------------------- Chatterbox
class ChatterboxEngine:
    THREADS = int(os.environ.get("CHATTERBOX_THREADS", "8") or 8)
    LANGUAGE = os.environ.get("CHATTERBOX_LANGUAGE", "de")

    def __init__(self) -> None:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        torch.set_num_threads(self.THREADS)
        self.torch = torch
        self.model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        self._default_conds = self.model.conds
        self._current = None

    def synthesize(self, job: dict):
        wav = job.get("ref_wav")
        key = (wav, os.path.getmtime(wav)) if wav else None
        if key != self._current:
            if wav:
                t0 = time.perf_counter()
                self.model.prepare_conditionals(wav, exaggeration=0.5)
                _log(f"Referenz {os.path.basename(wav)} vorbereitet in {time.perf_counter() - t0:.1f} s")
            else:
                self.model.conds = self._default_conds
            self._current = key
        with self.torch.inference_mode():
            out = self.model.generate(job["text"], language_id=self.LANGUAGE,
                                      exaggeration=float(job.get("exaggeration") or 0.5),
                                      cfg_weight=float(job.get("cfg_weight") or 0.5),
                                      temperature=float(job.get("temperature") or 0.8))
        return out.squeeze(0).cpu().numpy(), int(self.model.sr)


# --------------------------------------------------------------- VoxCPM2
class VoxCPMEngine:
    """OpenBMB VoxCPM2 (2B, Apache 2.0): 48 kHz, Klonen aus der Referenz
    ohne Abschrift möglich; mit Abschrift ("ultimate cloning") genauer."""
    THREADS = int(os.environ.get("VOXCPM_THREADS", "8") or 8)
    MODEL = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")
    STEPS = int(os.environ.get("VOXCPM_STEPS", "10") or 10)
    USE_TRANSCRIPT = os.environ.get("VOXCPM_USE_TRANSCRIPT", "1") not in ("0", "false", "no")

    def __init__(self) -> None:
        import inspect

        import torch
        from voxcpm import VoxCPM

        torch.set_num_threads(self.THREADS)
        kwargs = {"load_denoiser": False}
        # torch.compile bräuchte auf der CPU einen Compiler -> abschalten, falls möglich
        if "optimize" in inspect.signature(VoxCPM.from_pretrained).parameters:
            kwargs["optimize"] = False
        self.model = VoxCPM.from_pretrained(self.MODEL, **kwargs)

    def synthesize(self, job: dict):
        import numpy as np

        kw = {"text": job["text"], "cfg_value": 2.0, "inference_timesteps": self.STEPS}
        wav, txt = job.get("ref_wav"), job.get("ref_txt")
        if wav:
            if self.USE_TRANSCRIPT and txt and os.path.isfile(txt):
                kw.update(prompt_wav_path=wav, reference_wav_path=wav,
                          prompt_text=open(txt, encoding="utf-8").read().strip())
            else:
                kw["reference_wav_path"] = wav
        out = self.model.generate(**kw)
        return np.asarray(out, dtype=np.float32).reshape(-1), int(self.model.tts_model.sample_rate)


# ------------------------------------------------------------- OmniVoice
class OmniVoiceEngine:
    """k2-fsa OmniVoice (0.6B, Gewichte CC-BY-NC): Klonen mit Referenz +
    Abschrift, 24 kHz."""
    THREADS = int(os.environ.get("OMNIVOICE_THREADS", "8") or 8)
    MODEL = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")

    def __init__(self) -> None:
        import torch
        from omnivoice import OmniVoice

        torch.set_num_threads(self.THREADS)
        self.model = OmniVoice.from_pretrained(self.MODEL, device_map="cpu", dtype=torch.float32)

    def synthesize(self, job: dict):
        import numpy as np

        kw = {"text": job["text"]}
        wav, txt = job.get("ref_wav"), job.get("ref_txt")
        if wav:
            kw["ref_audio"] = wav
            if txt and os.path.isfile(txt):
                kw["ref_text"] = open(txt, encoding="utf-8").read().strip()
        audio = self.model.generate(**kw)
        return np.asarray(audio[0], dtype=np.float32).reshape(-1), 24000


ENGINES = {"chatterbox": ChatterboxEngine, "voxcpm": VoxCPMEngine, "omnivoice": OmniVoiceEngine}


def main() -> None:
    import numpy as np

    name = sys.argv[1]
    t0 = time.perf_counter()
    try:
        engine = ENGINES[name]()
    except Exception as exc:
        traceback.print_exc()
        _send("@@FAILED", {"error": f"{type(exc).__name__}: {exc}"})
        sys.exit(1)
    _log(f"{name}: Modell geladen in {time.perf_counter() - t0:.0f} s")
    _send("@@READY")
    tmpdir = os.environ.get("EXT_TMP", "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir())
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job = json.loads(line)
        try:
            t1 = time.perf_counter()
            samples, sr = engine.synthesize(job)
            compute = time.perf_counter() - t1
            fd, path = tempfile.mkstemp(suffix=".npy", dir=tmpdir)
            os.close(fd)
            np.save(path, np.asarray(samples, dtype=np.float32).reshape(-1))
            _send("@@RESULT", {"id": job["id"], "ok": True, "npy": path, "sr": sr,
                               "compute": compute, "load": 0.0})
        except Exception as exc:
            traceback.print_exc()
            _send("@@RESULT", {"id": job.get("id"), "ok": False,
                               "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
