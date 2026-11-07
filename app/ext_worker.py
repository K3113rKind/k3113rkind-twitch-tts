"""Rechenprozess für Engines mit eigener Python-Umgebung (NeuTTS, Chatterbox).

Läuft NICHT im App-Interpreter, sondern mit dem Python der jeweiligen venv
(/opt/neutts, /opt/chatterbox) – deren Abhängigkeiten (Torch-, Transformers-
Versionen) vertragen sich nicht mit Kokoro/Qwen/Pocket. Deshalb nur
Standardbibliothek + Engine-Pakete, keine Importe aus app/.

Protokoll (eine JSON-Zeile je Auftrag auf stdin):
  {"id": 1, "text": "...", "ref_wav": "/ref/x.wav", "ref_txt": "/ref/x.txt",
   "ref_codes": "/opt/neutts/samples/greta.pt"}
Antwort auf stdout (Bibliotheks-Ausgaben gehen nach stderr):
  @@RESULT {"id": 1, "ok": true, "npy": "/tmp/…npy", "sr": 24000,
            "compute": 1.2, "load": 0.0}
Beim Start wird das Modell sofort geladen und "@@READY" gemeldet.

Aufruf: /opt/neutts/bin/python app/ext_worker.py neutts
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


# ------------------------------------------------------------------ NeuTTS
class NeuTTSEngine:
    BACKBONE = os.environ.get("NEUTTS_BACKBONE", "neuphonic/neutts-nano-german-q4-gguf")
    CODEC = os.environ.get("NEUTTS_CODEC", "neuphonic/neucodec")
    THREADS = int(os.environ.get("NEUTTS_THREADS", "4") or 4)

    def __init__(self) -> None:
        import torch
        import neutts.neutts as nt

        torch.set_num_threads(self.THREADS)
        # NeuTTS nimmt sonst die Hälfte aller Kerne (bei 36 Kernen: 18) –
        # für ein 120M-Modell und mehrere Prozesse nebeneinander zu viel.
        nt._n_perf_cores = lambda: self.THREADS
        nt.os.cpu_count = lambda: self.THREADS
        self.torch = torch
        self.tts = nt.NeuTTS(backbone_repo=self.BACKBONE, backbone_device="cpu",
                             codec_repo=self.CODEC, codec_device="cpu")
        self._refs: dict[tuple, tuple] = {}

    def _reference(self, job: dict):
        codes_file = job.get("ref_codes")
        wav, txt = job.get("ref_wav"), job.get("ref_txt")
        src = codes_file or wav
        key = (src, os.path.getmtime(src), txt and os.path.getmtime(txt))
        if key not in self._refs:
            text = open(txt, encoding="utf-8").read().strip()
            if codes_file:
                codes = self.torch.load(codes_file, map_location="cpu")
            else:
                t0 = time.perf_counter()
                codes = self.tts.encode_reference(wav)
                _log(f"Referenz {os.path.basename(wav)} kodiert in {time.perf_counter() - t0:.1f} s")
            self._refs[key] = (codes, text)
        return self._refs[key]

    def synthesize(self, job: dict):
        codes, text = self._reference(job)
        wav = self.tts.infer(job["text"], codes, text,
                             temperature=float(job.get("temperature") or 1.0))
        return wav, 24000


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


ENGINES = {"neutts": NeuTTSEngine, "chatterbox": ChatterboxEngine}


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
