"""Rechenprozesse für Engines mit eigener Python-Umgebung (Chatterbox).

Jeder Prozess ist ein eigenständiges Programm (app/ext_worker.py) im Python
der jeweiligen venv. Gleiche Bedienung wie QwenPool: submit() wartet auf
einen freien Prozess und liefert (samples, Rechenzeit, Ladezeit, Decoder, sr).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import Future
from pathlib import Path

log = logging.getLogger(__name__)

WORKER = str(Path(__file__).with_name("ext_worker.py"))


class _Proc:
    def __init__(self, engine: str, idx: int, python: str) -> None:
        self.engine, self.idx = engine, idx
        env = {**os.environ, "HF_HUB_OFFLINE": "0", "PYTHONUNBUFFERED": "1"}
        if not env.get("HF_TOKEN"):
            env.pop("HF_TOKEN", None)
        self.p = subprocess.Popen([python, WORKER, engine], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, env=env, bufsize=1)
        self.ready = threading.Event()
        self.error: str | None = None
        self.futures: dict[int, Future] = {}
        threading.Thread(target=self._read_out, daemon=True,
                         name=f"{engine}-{idx + 1}-out").start()
        threading.Thread(target=self._read_err, daemon=True,
                         name=f"{engine}-{idx + 1}-err").start()

    def _read_out(self) -> None:
        for line in self.p.stdout:
            line = line.rstrip("\n")
            if line == "@@READY":
                self.ready.set()
            elif line.startswith("@@FAILED "):
                self.error = json.loads(line[9:]).get("error")
                self.ready.set()
            elif line.startswith("@@RESULT "):
                res = json.loads(line[9:])
                fut = self.futures.pop(res.get("id"), None)
                if fut is None:
                    continue
                if res.get("ok"):
                    import numpy as np

                    try:
                        samples = np.load(res["npy"])
                    finally:
                        Path(res["npy"]).unlink(missing_ok=True)
                    fut.set_result((samples, res["compute"], res.get("load", 0.0), 0.0, res["sr"]))
                else:
                    fut.set_exception(RuntimeError(res.get("error") or "Fehler im Rechenprozess"))
        # Prozess beendet
        self.error = self.error or f"Rechenprozess beendet (Code {self.p.poll()})"
        self.ready.set()
        for fut in list(self.futures.values()):
            if not fut.done():
                fut.set_exception(RuntimeError(f"{self.engine}: {self.error}"))
        self.futures.clear()

    def _read_err(self) -> None:
        for line in self.p.stderr:
            line = line.rstrip()
            if line:
                log.info("%s-%d: %s", self.engine, self.idx + 1, line[-400:])

    def alive(self) -> bool:
        return self.p.poll() is None

    def stop(self) -> None:
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.p.terminate()
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.p.kill()


class ExtPool:
    def __init__(self, engine: str, workers: int, python: str) -> None:
        self.engine, self.workers, self.python = engine, workers, python
        self._cond = threading.Condition()
        self._procs: list[_Proc | None] = [None] * workers
        self._idle: set[int] = set(range(workers))
        self._next_id = 0
        self._started = False
        self._closed = False

    def start(self) -> None:
        with self._cond:
            if self._started or self._closed:
                return
            for i in range(self.workers):
                self._procs[i] = _Proc(self.engine, i, self.python)
            self._started = True
        log.info("%s: %d Rechenprozesse gestartet", self.engine, self.workers)

    def shutdown(self) -> None:
        with self._cond:
            self._closed = True
            procs = list(self._procs)
            self._cond.notify_all()
        for proc in procs:
            if proc is not None:
                proc.stop()
        log.info("%s: Rechenprozesse beendet", self.engine)

    def _ensure(self, i: int) -> _Proc:
        proc = self._procs[i]
        if proc is None or (not proc.alive() and proc.ready.is_set()):
            if proc is not None:
                log.error("%s-Rechenprozess %d beendet (%s) – wird neu gestartet",
                          self.engine, i + 1, proc.error)
            proc = self._procs[i] = _Proc(self.engine, i, self.python)
        return proc

    def submit(self, args: dict, worker: int | None = None):
        """Blockierend (im Thread aufrufen)."""
        self.start()
        allowed = {worker} if worker is not None else set(range(self.workers))
        with self._cond:
            while not (self._idle & allowed):
                if self._closed:
                    raise RuntimeError("Rechenprozesse werden neu gestartet")
                self._cond.wait()
            if self._closed:
                raise RuntimeError("Rechenprozesse werden neu gestartet")
            i = min(self._idle & allowed)
            self._idle.discard(i)
            proc = self._ensure(i)
            jid = self._next_id
            self._next_id += 1
        try:
            # Erstes Laden (inkl. Download) kann dauern
            if not proc.ready.wait(timeout=1800):
                raise RuntimeError(f"{self.engine}: Modell lädt zu lange")
            if proc.error:
                raise RuntimeError(f"{self.engine}: {proc.error}")
            fut: Future = Future()
            proc.futures[jid] = fut
            t0 = time.perf_counter()
            proc.p.stdin.write(json.dumps({"id": jid, **args}) + "\n")
            proc.p.stdin.flush()
            result = fut.result()
            log.debug("%s: Auftrag in %.1f s", self.engine, time.perf_counter() - t0)
            return result
        finally:
            with self._cond:
                self._idle.add(i)
                self._cond.notify_all()
