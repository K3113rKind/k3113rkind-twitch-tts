"""Emote-Namen von Drittanbietern (BetterTTV, 7TV, FrankerFaceZ).

Twitch liefert nur die Positionen der *eigenen* Emotes in den
Nachrichten-Tags mit. Emotes von BTTV, 7TV oder FFZ stehen dort nicht drin –
sie sind für Twitch normaler Text und würden sonst buchstabiert vorgelesen
("KEKW", "OMEGALUL", "monkaS").

Deshalb holen wir die Namenslisten einmal pro Kanal von den öffentlichen
Schnittstellen der drei Dienste (keine Zugangsdaten nötig) und filtern
passende Wörter aus der Nachricht.

Die Kanal-ID kommt aus den IRC-Tags (`room-id`), es braucht also keine
Twitch-API-Anmeldung. Fehlschläge sind unkritisch: Klappt ein Abruf nicht
(offline, Dienst gestört), bleibt die Liste eben unvollständig.
"""

from __future__ import annotations

import asyncio
import logging
import time

log = logging.getLogger(__name__)

TIMEOUT = 10.0
CACHE_TTL = 6 * 3600  # Emote-Listen ändern sich selten; alle 6 Stunden neu

# room_id -> (Zeitstempel, Namen)
_cache: dict[str, tuple[float, frozenset[str]]] = {}
_locks: dict[str, asyncio.Lock] = {}


async def _get_json(client, url: str):
    try:
        resp = await client.get(url, timeout=TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        log.debug("Emote-Abruf %s: Status %s", url, resp.status_code)
    except Exception as exc:
        log.debug("Emote-Abruf %s fehlgeschlagen: %s", url, exc)
    return None


def _names_bttv(data) -> set[str]:
    names: set[str] = set()
    if isinstance(data, list):  # globale Liste
        entries = data
    elif isinstance(data, dict):  # Kanal: eigene + geteilte Emotes
        entries = (data.get("channelEmotes") or []) + (data.get("sharedEmotes") or [])
    else:
        return names
    for item in entries:
        if isinstance(item, dict) and item.get("code"):
            names.add(item["code"])
    return names


def _names_7tv(data) -> set[str]:
    names: set[str] = set()
    if not isinstance(data, dict):
        return names
    # Kanal: {"emote_set": {"emotes": [...]}}, global: {"emotes": [...]}
    emote_set = data.get("emote_set") if isinstance(data.get("emote_set"), dict) else data
    for item in emote_set.get("emotes") or []:
        if isinstance(item, dict) and item.get("name"):
            names.add(item["name"])
    return names


def _names_ffz(data) -> set[str]:
    names: set[str] = set()
    if not isinstance(data, dict):
        return names
    sets = data.get("sets")
    if not isinstance(sets, dict):
        return names
    for entry in sets.values():
        if not isinstance(entry, dict):
            continue
        for item in entry.get("emoticons") or []:
            if isinstance(item, dict) and item.get("name"):
                names.add(item["name"])
    return names


def cached_names(room_id: str) -> frozenset[str] | None:
    """Bereits geladene Namen zurückgeben, sonst None (ohne zu blockieren)."""
    cached = _cache.get(room_id)
    if cached and time.time() - cached[0] < CACHE_TTL:
        return cached[1]
    return None


async def fetch_names(room_id: str) -> frozenset[str]:
    """Emote-Namen für einen Kanal holen (mit Zwischenspeicher)."""
    now = time.time()
    cached = _cache.get(room_id)
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1]

    lock = _locks.setdefault(room_id, asyncio.Lock())
    async with lock:
        # Während des Wartens könnte ein anderer Aufruf schon geladen haben.
        cached = _cache.get(room_id)
        if cached and time.time() - cached[0] < CACHE_TTL:
            return cached[1]

        import httpx

        names: set[str] = set()
        async with httpx.AsyncClient(follow_redirects=True) as client:
            results = await asyncio.gather(
                _get_json(client, "https://api.betterttv.net/3/cached/emotes/global"),
                _get_json(client, f"https://api.betterttv.net/3/cached/users/twitch/{room_id}"),
                _get_json(client, "https://7tv.io/v3/emote-sets/global"),
                _get_json(client, f"https://7tv.io/v3/users/twitch/{room_id}"),
                _get_json(client, "https://api.frankerfacez.com/v1/set/global"),
                _get_json(client, f"https://api.frankerfacez.com/v1/room/id/{room_id}"),
            )

        bttv_global, bttv_room, sevtv_global, sevtv_room, ffz_global, ffz_room = results
        names |= _names_bttv(bttv_global)
        names |= _names_bttv(bttv_room)
        names |= _names_7tv(sevtv_global)
        names |= _names_7tv(sevtv_room)
        names |= _names_ffz(ffz_global)
        names |= _names_ffz(ffz_room)

        result = frozenset(names)
        _cache[room_id] = (time.time(), result)
        log.info("Emote-Listen für Kanal %s geladen: %d Namen", room_id, len(result))
        return result


def strip_names(text: str, names: frozenset[str]) -> str:
    """Wörter entfernen, die exakt einem Emote-Namen entsprechen.

    Bewusst Groß-/Kleinschreibung beachtend und nur ganze Wörter: So bleibt
    normaler Text unangetastet, auch wenn ein Emote zufällig wie ein
    gewöhnliches Wort heißt.
    """
    if not names:
        return text
    return " ".join(word for word in text.split() if word not in names)
