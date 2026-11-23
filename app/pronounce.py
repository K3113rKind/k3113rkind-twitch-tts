"""Aussprache-Liste und sprechbare Nutzernamen.

Aussprache-Liste (Einstellung "pronunciations", eine Regel pro Zeile):
    Mouzakrobat = Mausakrobat
    Gaming = Geeming
Ersetzt wird vor dem Vorlesen für alle Engines, ohne Rücksicht auf Groß-/
Kleinschreibung und nur ganze Wörter (Ziffern, _ und - zählen als Grenze,
"Mouzakrobat123" wird also auch erkannt). Zeilen mit # sind Kommentare.
"""

from __future__ import annotations

import re

CONFIG = None  # ConfigStore, wird in main gesetzt
MAX_RULES = 300

_LETTER = "A-Za-zÄÖÜäöüßÀ-ÿ"
_cache: tuple[str, list] = ("", [])


def parse(raw: str) -> list[tuple[str, str]]:
    rules = []
    for line in str(raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        word, _, say = line.partition("=")
        word, say = word.strip(), say.strip()
        if word and say and word.lower() != say.lower():
            rules.append((word, say))
    return rules[:MAX_RULES]


def _compiled(raw: str) -> list:
    global _cache
    if _cache[0] != raw:
        # längere Wörter zuerst ("Mouzakrobat" vor "Mouz")
        rules = sorted(parse(raw), key=lambda r: -len(r[0]))
        _cache = (raw, [(re.compile(rf"(?<![{_LETTER}]){re.escape(w)}(?![{_LETTER}])", re.I), s)
                        for w, s in rules])
    return _cache[1]


def apply(text: str, raw: str | None = None) -> str:
    if raw is None:
        raw = CONFIG.get("pronunciations") if CONFIG is not None else ""
    if not raw or not text:
        return text
    for rx, say in _compiled(raw):
        text = rx.sub(say, text)
    return text


_DECOR = re.compile(r"^(?:x+X+|X+x+)(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])(?:X+x+|x+X+)$")


def speakable_name(name: str) -> str:
    """Nutzernamen sprechbar machen: Aussprache-Liste, dann an _/-,
    Groß-Klein-Wechseln und Ziffern trennen ("MouzAkrobat" -> "Mouz Akrobat",
    "xXGamer99Xx" -> "Gamer 99")."""
    raw = name
    name = apply(name)
    if name != raw:
        return name  # eigene Regel geht vor
    n = _DECOR.sub("", name)
    n = re.sub(r"[_\-.]+", " ", n)
    n = re.sub(r"(?<=[a-zäöüß])(?=[A-ZÄÖÜ])", " ", n)
    # Ziffern nur am Wortende/-anfang abtrennen; mitten im Wort (Leetspeak
    # wie "K3113rKind") bleiben sie dran – dafür gibt es die Aussprache-Liste
    n = re.sub(rf"(?<=[{_LETTER}])(?=\d+(?:\s|$))|(?:(?<=^)|(?<=\s))(\d+)(?=[{_LETTER}])", r"\1 ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return apply(n) or raw
