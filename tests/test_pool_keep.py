"""Warmhalten mehrerer Modelle je Rechenprozess (ohne echte Modelle)."""
import queue
import threading

from app import qwen_pool
from app.qwen_pool import QwenPool


class FakeProc:
    def __init__(self, target):
        self.t = threading.Thread(target=target, daemon=True)
        self.t.start()
        self.pid = 0
        self.exitcode = None

    def is_alive(self):
        return self.t.is_alive()

    def join(self, timeout=None):
        self.t.join(timeout)

    def terminate(self):
        pass

    kill = terminate


def make_pool(workers, keep, monkeypatch, budget=1000.0):
    monkeypatch.setattr(qwen_pool, "ram_budget_gb", lambda: budget)
    monkeypatch.setattr(qwen_pool, "_mem_available_gb", lambda: None)
    pool = QwenPool(workers, 4, engine="qwen")
    pool.keep = keep
    spawns = []

    def spawn(i):
        q = queue.Queue()
        out = pool._outq

        def loop():
            while True:
                job = q.get()
                if job is None:
                    return
                jid, args = job
                out.put((jid, True, ("ok", 0.0, 0.0, 0.0)))

        spawns.append(i)
        pool._shares[i] = None
        pool._inqs[i], pool._procs[i] = q, FakeProc(loop)

    pool._spawn = spawn
    return pool, spawns


A, B, C = "Qwen/x-1.7B-Base", "Qwen/x-0.6B-Base", "Qwen/x-1.7B-CustomVoice"


def test_keep_two_models_no_recycle(monkeypatch):
    pool, spawns = make_pool(1, 2, monkeypatch)
    for m in (A, B, A, B):
        assert pool.submit({"repo": m, "text": "x"}) == ("ok", 0.0, 0.0, 0.0)
    assert spawns == [0]  # nie neu gestartet
    assert pool._loaded[0] == [A, B]
    pool.submit({"repo": C, "text": "x"})  # drittes Modell -> Neustart
    assert spawns == [0, 0] and pool._loaded[0] == [C]
    pool.shutdown()


def test_keep_one_like_before(monkeypatch):
    pool, spawns = make_pool(1, 1, monkeypatch)
    pool.submit({"repo": A, "text": "x"})
    pool.submit({"repo": B, "text": "x"})
    assert spawns == [0, 0] and pool._loaded[0] == [B]
    pool.shutdown()


def test_ram_guard_prevents_stacking(monkeypatch):
    pool, spawns = make_pool(1, 3, monkeypatch, budget=15.0)  # 10 + 10 GB passt nicht
    pool.submit({"repo": A, "text": "x"})
    pool.submit({"repo": C, "text": "x"})
    assert spawns == [0, 0] and pool._loaded[0] == [C]
    pool.shutdown()


def test_pin_never_evicts(monkeypatch):
    pool, spawns = make_pool(1, 2, monkeypatch, budget=15.0)
    pool.pinned = A
    pool.submit({"repo": C, "text": "x"})
    assert pool.submit({"repo": A, "text": "x", "_pin": True}) is None  # passt nicht -> nichts tun
    assert pool._loaded[0] == [C] and spawns == [0]
    pool.shutdown()
