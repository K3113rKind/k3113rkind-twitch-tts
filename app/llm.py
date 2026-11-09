"""Lokales Sprachmodell im LAN (OpenAI-kompatibel: llama.cpp, Ollama, …).

Eingestellt über die .env (gemeinsam für Chat-Aufbereitung und YouTube):
  LLM_URL / YT_LLM_URL      z. B. http://192.168.0.190:8080/v1
  LLM_MODEL / YT_LLM_MODEL  Modellname wie unter /v1/models
  LLM_KEY / YT_LLM_KEY      optional
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from collections import OrderedDict

log = logging.getLogger(__name__)

URL = (os.environ.get("LLM_URL") or os.environ.get("YT_LLM_URL") or "").rstrip("/")
MODEL = os.environ.get("LLM_MODEL") or os.environ.get("YT_LLM_MODEL") or ""
KEY = os.environ.get("LLM_KEY") or os.environ.get("YT_LLM_KEY") or ""
CHAT_TIMEOUT_S = float(os.environ.get("CHAT_LLM_TIMEOUT", "6") or 6)


def available() -> bool:
    return bool(URL)


def complete(messages: list[dict], timeout: float, temperature: float = 0.3, max_tokens: int | None = None) -> str:
    body = {"model": MODEL or "default", "messages": messages, "temperature": temperature,
            "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    if max_tokens:
        body["max_tokens"] = max_tokens
    req = urllib.request.Request(f"{URL}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {KEY}"} if KEY else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    text = data["choices"][0]["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


# ------------------------------------------------- Chat-Nachrichten aufbereiten
CHAT_PROMPT = (
    "Du bereitest Twitch-Chat-Nachrichten für eine deutsche Sprachausgabe auf. Gib die Nachricht "
    "so zurück, wie ein Mensch sie natürlich vorlesen würde:\n"
    "- Ist sie nicht auf Deutsch, übersetze sie sinngemäß ins Deutsche.\n"
    "- Chat-Kürzel und Slang in sprechbare Wörter (lol -> haha, gg -> gut gespielt, "
    "xD -> haha, w8 -> warte, ka -> keine Ahnung), Zahlen, Uhrzeiten und Einheiten ausschreiben.\n"
    "- Emojis und Zeichensalat weglassen, außer sie tragen die Aussage (dann kurz in Worten).\n"
    "- Inhalt, Ton und Wortwahl sonst NICHT ändern: nichts hinzufügen, nichts erklären, nichts "
    "beschönigen oder zensieren, keine Anrede ergänzen.\n"
    "- Antworte NUR mit dem Text zum Vorlesen, ohne Anführungszeichen."
)
_cache: OrderedDict[str, str] = OrderedDict()
_warned = False


def prepare_chat(text: str) -> str:
    """Blockierend (im Thread aufrufen). Bei Fehler/Zeitüberschreitung: Original."""
    global _warned
    if not URL or not text.strip():
        return text
    if text in _cache:
        _cache.move_to_end(text)
        return _cache[text]
    try:
        out = complete([{"role": "system", "content": CHAT_PROMPT}, {"role": "user", "content": text}],
                       timeout=CHAT_TIMEOUT_S, temperature=0.2, max_tokens=200)
    except Exception as exc:
        if not _warned:
            log.warning("Chat-LLM nicht nutzbar (%s) – lese Nachrichten unverändert", exc)
            _warned = True
        return text
    _warned = False
    out = out.strip().strip('"„“').strip()
    # Plausibilität: leer oder ausufernd (Modell plaudert) -> Original
    if not out or len(out) > max(80, 3 * len(text)):
        return text
    _cache[text] = out
    while len(_cache) > 500:
        _cache.popitem(last=False)
    return out
