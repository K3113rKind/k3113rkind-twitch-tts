"""Persistente Konfiguration in einer gemounteten Datei.

Wird beim Start geladen und bei jeder Änderung in der GUI sofort
gespeichert (atomar per Temp-Datei + rename), überlebt also
Container-Updates.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from .voices import DEFAULT_VOICE, QWEN_STYLES, VOICES, sync_clone_voices

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "channel": "",
    "voice": DEFAULT_VOICE,
    "volume": 1.0,          # 0.0-2.0, als Gain im Audio-Stream angewandt
    "speed": 1.0,           # Kokoro-Speed-Parameter
    "read_username": True,  # Namen vor der Nachricht vorlesen
    # Wie der Name angekündigt wird: "doppelpunkt" -> "Peter: hallo",
    # "sagt" -> "Peter sagt hallo" (bei englischen Stimmen "says").
    "username_style": "doppelpunkt",
    "read_emotes": False,   # Emote-Codes mitlesen statt entfernen
    "read_mentions": True,  # Nachrichten mit @Erwähnung vorlesen
    "read_smileys": True,   # Text-Smileys und Emojis mitlesen
    # Regelmäßiger kurzer Impuls, damit Lautsprecher/Bluetooth-Boxen bei
    # längerer Stille nicht in den Standby gehen.
    "keep_speakers_awake": True,
    # Pegel des durchgehenden Hintergrundtons. Zweck: Der Browser stuft den
    # Tab dann als tonausgebend ein und drosselt ihn im Hintergrund nicht.
    # Die nötige Stärke ist je nach Browser verschieden, deshalb einstellbar.
    "keepalive_level": 0.003,
    # Qwen: wie stark die Betonung variiert (Sampling-Temperatur). Niedriger =
    # gleichmäßiger und näher an der Referenz, höher = lebendiger, aber wilder.
    "qwen_temperature": 0.9,
    # Sprechstil der Qwen-Preset-Stimmen (Schlüssel aus voices.QWEN_STYLES
    # oder "eigen" mit freiem Text in qwen_style_custom)
    "qwen_style": "neutral",
    "qwen_style_custom": "",
    # Eigene Anzeigenamen für Stimmen (Stimmen-Schlüssel -> Name), in der GUI
    # über "Umbenennen" gesetzt; leerer Name = Standardname.
    "voice_labels": {},
    # Sichtbarkeit in der Stimmauswahl (Stimmen-Schlüssel -> an/aus), gesetzt
    # über "Stimmen verwalten" bzw. "Aus der Liste entfernen". Nicht eingetragene
    # Stimmen folgen ihrem Standard (Katalogstimmen aus, alle anderen an).
    "voice_visibility": {},
    # Rechenprozesse je Engine (1–6), in der GUI unter "Rechenprozesse".
    # Startwert Qwen aus QWEN_WORKERS (älteres Setup über .env).
    # Freigabe für die stabile Instanz ("Freunde"): nur diese Stimmen sind über
    # /api/share/* nutzbar, geschützt mit share_token (wird beim ersten Start
    # erzeugt, in "Stimmen verwalten" angezeigt und neu erzeugbar).
    "shared_voices": [],
    "share_token": "",
    "workers": {"qwen": int(os.environ.get("QWEN_WORKERS", "1") or 1), "pocket": 1,
                "neutts": 1, "chatterbox": 1},
    "cooldown_seconds": 10, # Mindestabstand zwischen zwei Nachrichten je User
    "queue_limit": 10,      # max. wartende Nachrichten; ältere werden verworfen
    "bot_blocklist": [
        "nightbot",
        "streamelements",
        "streamlabs",
        "moobot",
        "fossabot",
    ],
}


_TWITCH_URL = re.compile(r"twitch\.tv/(?:(?:popout|moderator|embed)/)?([A-Za-z0-9_]{2,25})", re.I)


def clean_channel(value) -> str:
    """Kanalname aus Eingabe holen: auch ganze Links wie
    https://www.twitch.tv/name, twitch.tv/popout/name/chat, @name oder #name."""
    value = str(value or "").strip()
    match = _TWITCH_URL.search(value)
    if match:
        value = match.group(1)
    return value.lstrip("@#").strip().lower()


class ConfigStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or os.environ.get("CONFIG_PATH", "/config/config.json"))
        self._lock = threading.Lock()
        self._data: dict[str, Any] = dict(DEFAULTS)
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            self._data = {**DEFAULTS, **{k: v for k, v in loaded.items() if k in DEFAULTS}}
            log.info("Konfiguration geladen: %s", self.path)
            if not self._data.get("share_token"):
                self._data["share_token"] = secrets.token_urlsafe(24)
                self.save()
        except FileNotFoundError:
            log.info("Keine Konfiguration gefunden, schreibe Defaults nach %s", self.path)
            self._data["share_token"] = secrets.token_urlsafe(24)
            self.save()
        except (json.JSONDecodeError, OSError) as exc:
            log.error("Konfiguration unlesbar (%s) – nutze Defaults", exc)

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(self._data, fh, indent=2, ensure_ascii=False)
                # mkstemp legt 0600 an; lesbar lassen, damit die Datei auf dem
                # Host nicht nur für root zugänglich ist (Build, Backups, Editor).
                os.chmod(tmp, 0o644)
                os.replace(tmp, self.path)
            except OSError:
                try:
                    os.unlink(tmp)
                finally:
                    raise

    def get(self, key: str) -> Any:
        return self._data[key]

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in changes.items():
            if key not in DEFAULTS:
                continue
            if key in ("cooldown_seconds", "queue_limit"):
                value = max(0, int(value))
                if key == "queue_limit":
                    value = max(1, value)
            elif key == "qwen_style":
                value = str(value).strip()
                if value not in QWEN_STYLES and value != "eigen":
                    continue
            elif key == "qwen_style_custom":
                value = str(value).strip()[:300]
            elif key == "qwen_temperature":
                value = round(min(max(float(value), 0.3), 1.2), 2)
            elif key == "keepalive_level":
                value = min(max(float(value), 0.0), 0.05)
            elif key in ("volume", "speed"):
                value = min(max(float(value), 0.0), 3.0)
            elif key in ("read_username", "read_emotes", "read_mentions", "read_smileys",
                         "keep_speakers_awake"):
                value = bool(value)
            elif key == "voice_labels":
                if not isinstance(value, dict):
                    continue
                value = {
                    str(k): str(v).strip()[:80]
                    for k, v in value.items()
                    if str(v).strip()
                }
            elif key == "workers":
                if not isinstance(value, dict):
                    continue
                value = {**self._data.get("workers", {}),
                         **{str(k): min(max(int(v), 1), 6) for k, v in value.items()
                            if str(k) in ("qwen", "pocket", "neutts", "chatterbox")}}
            elif key == "shared_voices":
                if not isinstance(value, list):
                    continue
                value = sorted({str(v) for v in value})
            elif key == "share_token":
                value = str(value).strip()
                if len(value) < 16:
                    continue
            elif key == "voice_visibility":
                if not isinstance(value, dict):
                    continue
                value = {str(k): bool(v) for k, v in value.items()}
            elif key == "bot_blocklist":
                value = sorted({str(v).strip().lower() for v in value if str(v).strip()})
            elif key == "channel":
                value = clean_channel(value)
            elif key == "username_style":
                value = str(value).strip()
                if value not in ("doppelpunkt", "sagt"):
                    continue
            elif key == "voice":
                value = str(value).strip()
                sync_clone_voices()
                if value not in VOICES:
                    continue  # unbekannte Stimme ignorieren
            else:
                value = str(value).strip()
            cleaned[key] = value
        self._data.update(cleaned)
        self.save()
        return self.as_dict()
