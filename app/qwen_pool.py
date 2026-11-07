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

Einschalten: in der Oberfläche unter "Rechenprozesse" (je Engine 1–6) oder
als Startwert QWEN_WORKERS=4 (Standard 1 = aus, alles im Hauptprozess).
Speicher: je Prozess ca. 3 GB (0.6B) bzw. 7–8 GB (1.7B). Große Modelle
laufen deshalb nur auf QWEN_WORKERS_BIG Prozessen (Standard: automatisch
nach Arbeitsspeicher, ~8 GB je Prozess).

Dieselbe Mechanik gibt es auch für Pocket TTS (engine="pocket"): Ein
Prozess rechnet dort mit POCKET_THREADS Threads, mehrere Prozesse rechnen
mehrere Nachrichten gleichzeitig.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
import contextvars
from concurrent.futures import Future

# Reihenfolge der Aufträge: Wer wartet, kommt nach PRIORITY dran (kleiner =
# früher), sonst nach Eingang. Der Player setzt (Eingang der Nachricht,
# Abschnittsnummer) – so rechnet ein frei werdender Prozess immer den als
# Nächstes benötigten Satz. asyncio.to_thread übernimmt den Wert in den Thread.
PRIORITY: contextvars.ContextVar = contextvars.ContextVar("tts_priority", default=None)


def with_priority(prio, fn, *args, **kwargs):
    """fn im aktuellen Thread mit gesetzter Priorität aufrufen."""
    PRIORITY.set(prio)
    return fn(*args, **kwargs)

log = logging.getLogger(__name__)

QWEN_WORKERS = int(os.environ.get("QWEN_WORKERS", "1") or 1)
def _mem_total_gb() -> float:
    """Arbeitsspeicher, den der Container sieht (im LXC: dessen Grenze)."""
    try:
        limit = open("/sys/fs/cgroup/memory.max").read().strip()
        if limit.isdigit():
            return int(limit) / 2**30
    except OSError:
        pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


def _big_workers_default() -> int:
    """Wie viele Prozesse dürfen 1.7B laden? ~8 GB je Prozess, ~6 GB Reserve
    für App, Kokoro & Co. 24 GB -> 2, 32 GB -> 3, 64 GB -> 7."""
    gb = _mem_total_gb()
    return max(1, int((gb - 6) // 8)) if gb else 2


# Leer/0 = automatisch nach Arbeitsspeicher
QWEN_WORKERS_BIG = int(os.environ.get("QWEN_WORKERS_BIG", "0") or 0) or _big_workers_default()


def _worker_main(idx: int, inq, outq, share, engine_name: str = "qwen") -> None:
    """Läuft im eigenen Prozess: Anfragen holen, rechnen, Ergebnis zurück.
    share: vom Hauptprozess laufend angepasster Thread-Anteil – wird vor
    jedem Rechenschritt übernommen (live umverteilen, wenn Nachrichten
    dazukommen oder fertig werden)."""
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s %(levelname)s {engine_name}-{idx + 1} %(name)s: %(message)s")
    if engine_name == "pocket":
        _pocket_worker(inq, outq)
        return
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


def _pocket_worker(inq, outq) -> None:
    """Pocket TTS in einem eigenen Prozess."""
    from .pocket import PocketEngine

    engine = PocketEngine()
    while True:
        job = inq.get()
        if job is None:
            return
        jid, args = job
        try:
            t0 = time.perf_counter()
            engine.load(args["language"])
            load_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            samples, sr = engine.synthesize(args["text"], args["language"], args.get("voice"),
                                            args.get("ref_wav"))
            outq.put((jid, True, (samples, time.perf_counter() - t1, load_s, 0.0, sr)))
        except Exception as exc:
            outq.put((jid, False, f"{type(exc).__name__}: {exc}"))


class QwenPool:
    """Mehrere Rechenprozesse für eine Engine ("qwen" oder "pocket")."""

    def __init__(self, workers: int, total_threads: int, engine: str = "qwen") -> None:
        self.engine = engine
        self._closed = False
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
        self._waiting: list[tuple] = []  # (Priorität, Ticket, erlaubte Prozesse)
        self._ticket = 0
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
        threading.Thread(target=self._reader, name=f"{self.engine}-pool-reader", daemon=True).start()
        log.info("%s: %d Rechenprozesse gestartet (zusammen %d Threads, 1.7B auf bis zu %d, RAM %.0f GB)",
                 self.engine, self.workers, self.total_threads,
                 min(self.workers, QWEN_WORKERS_BIG), _mem_total_gb())

    def shutdown(self) -> None:
        """Alle Prozesse beenden (z. B. vor dem Umstellen der Anzahl). Laufende
        Anfragen enden mit Fehler, wartende werden abgewiesen."""
        with self._cond:
            self._closed = True
            for jid, (fut, _wi) in list(self._futures.items()):
                self._futures.pop(jid, None)
                if not fut.done():
                    fut.set_exception(RuntimeError("Rechenprozesse werden neu gestartet"))
            self._cond.notify_all()
            procs, inqs = list(self._procs), list(self._inqs)
        for q in inqs:
            try:
                if q is not None:
                    q.put(None)
            except Exception:
                pass
        for p in procs:
            if p is None:
                continue
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        log.info("%s: %d Rechenprozesse beendet", self.engine, self.workers)

    def _spawn(self, i: int) -> None:
        q = self._ctx.Queue()
        share = self._ctx.Value("i", self.total_threads, lock=False)
        self._shares[i] = share
        p = self._ctx.Process(target=_worker_main, args=(i, q, self._outq, share, self.engine),
                              name=f"{self.engine}-{i + 1}", daemon=True)
        p.start()
        self._inqs[i], self._procs[i] = q, p

    def _reader(self) -> None:
        while not self._closed:
            try:
                jid, ok, payload = self._outq.get(timeout=2)
            except queue.Empty:
                self._check_dead()
                continue
            except (EOFError, OSError):
                return
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
            if self._closed:
                return
            for i, p in enumerate(self._procs):
                if p is None or p.is_alive():
                    continue
                log.error("%s-Rechenprozess %d beendet (Code %s) – wird neu gestartet",
                          self.engine, i + 1, p.exitcode)
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
    def _allowed(self, repo: str | None) -> set[int]:
        if repo and "1.7B" in repo:
            return set(range(min(self.workers, max(1, QWEN_WORKERS_BIG))))
        return set(range(self.workers))

    def submit(self, args: dict, worker: int | None = None):
        """Blockierend (im Thread aufrufen): wartet auf einen freien Prozess,
        rechnet dort und liefert (samples, Rechenzeit, Ladezeit, Decoder)."""
        self.start()
        allowed = {worker} if worker is not None else self._allowed(args.get("repo"))
        prio = PRIORITY.get()
        with self._cond:
            self._ticket += 1
            me = ((prio if prio is not None else (time.monotonic(), 0)), self._ticket, frozenset(allowed))
            self._waiting.append(me)
            try:
                # Feste Reihenfolge: dran ist der wartende Auftrag mit der
                # kleinsten Priorität, für den gerade ein passender Prozess frei ist
                while True:
                    if self._closed:
                        raise RuntimeError("Rechenprozesse werden neu gestartet")
                    servable = [w for w in self._waiting if self._idle & w[2]]
                    if servable and min(servable) is me:
                        break
                    self._cond.wait()
            finally:
                self._waiting.remove(me)
                self._cond.notify_all()
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
