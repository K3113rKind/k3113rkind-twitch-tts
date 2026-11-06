"""Mehrere Qwen-Rechenprozesse parallel (nur CPU).

Messung R740 (Sockel 0, 18 Kerne, Qwen 0.6B Base, candy-lang):
  1 Prozess  18 Threads: RTF 2,7  -> Durchsatz 0,37x Echtzeit
  3 Prozesse je 6:       RTF 4,4  -> Durchsatz 0,68x
  4 Prozesse je 4:       RTF 5,3  -> Durchsatz 0,75x
Die Schritte von Qwen sind auf der CPU klein (Verwaltungsaufwand statt
Rechenleistung) – ein Prozess lastet 18 Kerne nicht aus. Getrennte Prozesse
umgehen außerdem die Python-Sperre (GIL), die Threads im selben Prozess
ausbremsen würde.

Jede Anfrage geht an einen freien Prozess. Die Threads werden laufend nach
Last verteilt: rechnet ein Prozess allein, bekommt er alle QWEN_THREADS;
kommen weitere dazu, geben die laufenden beim nächsten Rechenschritt ab
(und holen sich die Threads zurück, sobald andere fertig sind).
Stürzt ein Prozess ab (z. B. Speicher), wird seine Anfrage mit Fehler
beendet und der Prozess neu gestartet.

Einschalten: QWEN_WORKERS=4 (Standard 1 = aus, alles im Hauptprozess).
Speicher: je Prozess ca. 3 GB (0.6B) bzw. 7–8 GB (1.7B). Große Modelle
laufen deshalb nur auf QWEN_WORKERS_BIG Prozessen (Standard 2).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent.futures import Future

log = logging.getLogger(__name__)

QWEN_WORKERS = int(os.environ.get("QWEN_WORKERS", "1") or 1)
QWEN_WORKERS_BIG = int(os.environ.get("QWEN_WORKERS_BIG", "2") or 2)


def _worker_main(idx: int, inq, outq, share) -> None:
    """Läuft im eigenen Prozess: Anfragen holen, rechnen, Ergebnis zurück.
    share: vom Hauptprozess laufend angepasster Thread-Anteil – wird vor
    jedem Rechenschritt übernommen (live umverteilen, wenn Nachrichten
    dazukommen oder fertig werden)."""
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s %(levelname)s qwen-{idx + 1} %(name)s: %(message)s")
    import torch

    from .tts import QwenEngine

    engine = QwenEngine()
    current = [0]
    wrapped = [None]

    def follow_share() -> None:
        n = share.value
        if n != current[0]:
            torch.set_num_threads(n)
            current[0] = n

    def wrap_model() -> None:
        """Vor jedem Schritt von Talker und Code-Predictor den Anteil prüfen."""
        model = engine._model
        if model is None or wrapped[0] is model:
            return
        talker = model.model.talker
        for mod in (talker.model, talker.code_predictor.model):
            original = mod.forward

            def forward(*a, _orig=original, **k):
                follow_share()
                return _orig(*a, **k)

            mod.forward = forward
        wrapped[0] = model
    while True:
        job = inq.get()
        if job is None:
            return
        jid, args = job
        try:
            t0 = time.perf_counter()
            engine.load(args["repo"])
            wrap_model()
            load_s = time.perf_counter() - t0
            follow_share()
            t1 = time.perf_counter()
            samples = engine.synthesize(
                args["text"], args["repo"], args.get("speaker"), args.get("ref_wav"),
                args.get("ref_txt"), args.get("temperature"), args.get("instruct"),
            )
            outq.put((jid, True, (samples, time.perf_counter() - t1, load_s,
                                  engine.last_decode_s)))
        except Exception as exc:  # Fehler zurückmelden statt Prozess sterben lassen
            outq.put((jid, False, f"{type(exc).__name__}: {exc}"))


class QwenPool:
    def __init__(self, workers: int, total_threads: int) -> None:
        self.workers = workers
        self.total_threads = max(1, total_threads)
        self._ctx = mp.get_context("spawn")  # kein fork: Torch/Threads im Elternprozess
        self._cond = threading.Condition()
        self._idle: set[int] = set(range(workers))
        self._procs: list = [None] * workers
        self._inqs: list = [None] * workers
        self._shares: list = [None] * workers  # Thread-Anteil je Prozess (geteilt)
        self._outq = None
        self._futures: dict[int, tuple[Future, int]] = {}
        self._next_id = 0
        self._started = False

    # ------------------------------------------------------------ Verwaltung
    def start(self) -> None:
        with self._cond:
            if self._started:
                return
            self._outq = self._ctx.Queue()
            for i in range(self.workers):
                self._spawn(i)
            self._started = True
        threading.Thread(target=self._reader, name="qwen-pool-reader", daemon=True).start()
        log.info("Qwen: %d Rechenprozesse gestartet (zusammen %d Threads)",
                 self.workers, self.total_threads)

    def _spawn(self, i: int) -> None:
        q = self._ctx.Queue()
        share = self._ctx.Value("i", self.total_threads, lock=False)
        self._shares[i] = share
        p = self._ctx.Process(target=_worker_main, args=(i, q, self._outq, share),
                              name=f"qwen-{i + 1}", daemon=True)
        p.start()
        self._inqs[i], self._procs[i] = q, p

    def _reader(self) -> None:
        while True:
            try:
                jid, ok, payload = self._outq.get(timeout=2)
            except queue.Empty:
                self._check_dead()
                continue
            with self._cond:
                entry = self._futures.pop(jid, None)
            if entry is None:
                continue
            fut = entry[0]
            if ok:
                fut.set_result(payload)
            else:
                fut.set_exception(RuntimeError(payload))

    def _check_dead(self) -> None:
        with self._cond:
            for i, p in enumerate(self._procs):
                if p is None or p.is_alive():
                    continue
                log.error("Qwen-Rechenprozess %d beendet (Code %s) – wird neu gestartet",
                          i + 1, p.exitcode)
                for jid, (fut, wi) in list(self._futures.items()):
                    if wi == i:
                        self._futures.pop(jid, None)
                        fut.set_exception(RuntimeError(
                            f"Rechenprozess {i + 1} abgestürzt (zu wenig Speicher?)"))
                self._spawn(i)

    def _rebalance(self) -> None:
        """Threads auf die gerade rechnenden Prozesse verteilen (unter _cond)."""
        busy = [i for i in range(self.workers) if i not in self._idle]
        if not busy:
            return
        share = max(2, self.total_threads // len(busy))
        for i in busy:
            if self._shares[i] is not None:
                self._shares[i].value = share

    # ------------------------------------------------------------- Anfragen
    def _allowed(self, repo: str) -> set[int]:
        if "1.7B" in repo:
            return set(range(min(self.workers, max(1, QWEN_WORKERS_BIG))))
        return set(range(self.workers))

    def submit(self, args: dict, worker: int | None = None):
        """Blockierend (im Thread aufrufen): wartet auf einen freien Prozess,
        rechnet dort und liefert (samples, Rechenzeit, Ladezeit, Decoder)."""
        self.start()
        allowed = {worker} if worker is not None else self._allowed(args["repo"])
        with self._cond:
            while not (self._idle & allowed):
                self._cond.wait()
            i = min(self._idle & allowed)
            self._idle.discard(i)
            self._rebalance()
            jid = self._next_id
            self._next_id += 1
            fut: Future = Future()
            self._futures[jid] = (fut, i)
        try:
            self._inqs[i].put((jid, dict(args)))
            return fut.result()
        finally:
            with self._cond:
                self._idle.add(i)
                self._rebalance()
                self._cond.notify_all()
