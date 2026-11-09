"""Vergabe an die Rechenprozesse: strikt nach Priorität (Lesereihenfolge),
auch wenn alle Abschnitte gleichzeitig ankommen."""
import random
import threading
import time

import app.qwen_pool as qp


def _stub_pool(workers=4):
    p = object.__new__(qp.QwenPool)
    p.engine = "qwen"
    p.workers = workers
    p._cond = threading.Condition()
    p._waiting, p._ticket, p._idle, p._closed = [], 0, set(range(workers)), False
    p._futures, p._next_id = {}, 0
    p._loaded, p._recycling = [None] * workers, set()
    started = []

    class Q:
        def put(self, item):
            if item is None:
                return
            jid, args = item
            started.append(args["n"])
            fut, _ = p._futures[jid]
            threading.Timer(0.03 + random.random() * 0.02, lambda: fut.set_result(args["n"])).start()

    p._inqs = [Q() for _ in range(workers)]
    p._procs = [None] * workers
    p.start = lambda: None
    p._allowed = lambda repo: set(range(workers))
    p._rebalance = lambda: None
    return p, started


def test_first_wave_is_first_chunks():
    p, started = _stub_pool(4)

    def job(n):
        time.sleep(random.random() * 0.01)
        qp.PRIORITY.set((1.0, n))
        p.submit({"n": n, "repo": "m"})

    threads = [threading.Thread(target=job, args=(n,)) for n in random.sample(range(17), 17)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(started[:4]) == [0, 1, 2, 3]
    assert started[4:] == sorted(started[4:])


def test_model_switch_restarts_worker():
    p, _ = _stub_pool(2)
    spawned = []

    class P:
        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

    p._procs = [P(), P()]
    p._spawn = lambda i: (spawned.append(i), p._procs.__setitem__(i, P()))
    p._loaded = ["1.7B", "1.7B"]
    for w in (0, 1):
        p.submit({"n": w, "repo": "0.6B"}, worker=w)
    assert sorted(spawned) == [0, 1]
    assert p._loaded == ["0.6B", "0.6B"]
