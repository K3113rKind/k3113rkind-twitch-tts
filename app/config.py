"""Persistente Konfiguration in einer gemounteten Datei.

Wird beim Start geladen und bei jeder Änderung in der GUI sofort
gespeichert (atomar per Temp-Datei + rename), überlebt also
Container-Updates.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from .voices import DEFAULT_VOICE, VOICES

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "channel": "",
    # Optionales Twitch-OAuth-Token. Wird nur gespeichert (für spätere
    # Funktionen wie Antworten senden); die Chat-Verbindung selbst ist
    # anonym/read-only.
    "oauth_token": "",
    "voice": DEFAULT_VOICE,
    "volume": 1.0,          # 0.0-2.0, als Gain im Browser angewandt
    "speed": 1.0,           # Kokoro-Speed-Parameter
    "read_username": True,  # "<user>: <nachricht>" vorlesen
    "read_emotes": False,   # Emote-Codes mitlesen statt entfernen
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
        except FileNotFoundError:
            log.info("Keine Konfiguration gefunden, schreibe Defaults nach %s", self.path)
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
            elif key in ("volume", "speed"):
                value = min(max(float(value), 0.0), 3.0)
            elif key in ("read_username", "read_emotes"):
                value = bool(value)
            elif key == "bot_blocklist":
                value = sorted({str(v).strip().lower() for v in value if str(v).strip()})
            elif key == "channel":
                value = str(value).strip().lstrip("#").lower()
            elif key == "voice":
                value = str(value).strip()
                if value not in VOICES:
                    continue  # unbekannte Stimme ignorieren
            else:
                value = str(value).strip()
            cleaned[key] = value
        self._data.update(cleaned)
        self.save()
        return self.as_dict()
