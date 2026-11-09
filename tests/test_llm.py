"""Chat-Aufbereitung: Rückfall aufs Original bei Fehlern/Unsinn."""
import app.llm as llm


def test_chat_llm_uses_answer(monkeypatch):
    monkeypatch.setattr(llm, "URL", "http://x")
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "Haha, Bruder, das war krass.")
    llm._cache.clear()
    assert llm.prepare_chat("lol bro dat war sick") == "Haha, Bruder, das war krass."


def test_chat_llm_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "URL", "http://x")
    llm._cache.clear()

    def boom(*a, **k):
        raise TimeoutError("zu langsam")
    monkeypatch.setattr(llm, "complete", boom)
    assert llm.prepare_chat("hallo") == "hallo"
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "Gerne erkläre ich dir ausführlich, " * 20)
    assert llm.prepare_chat("hi") == "hi"


def test_chat_llm_off_without_url(monkeypatch):
    monkeypatch.setattr(llm, "URL", "")
    assert llm.prepare_chat("lol") == "lol"
