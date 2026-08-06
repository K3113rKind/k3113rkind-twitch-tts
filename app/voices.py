"""Feste Stimmen-Liste – die einzige Stelle, an der Modelldateien vorkommen.

Bewusst hartkodiert: Diese Version soll ohne Modell-, Engine- oder
Download-Auswahl auskommen. Der Nutzer sieht nur Klarnamen ("Victoria"),
nie Dateinamen oder Backbone-Begriffe.

Kokoro-Namenskonvention beim Voicepack: erster Buchstabe = Sprache
(d = Deutsch, a = American English), zweiter = Geschlecht. Der Sprachcode
muss zum Backbone passen – deutsche Stimmen laufen auf den kikiri-
Fine-Tunes, englische auf dem offiziellen Kokoro-82M.
"""

from __future__ import annotations

import os
from pathlib import Path

# Reihenfolge = Reihenfolge im Dropdown.
VOICES: dict[str, dict] = {
    "victoria": {
        "label": "Victoria – Deutsch, weiblich",
        "backbone": "kikiri_german_victoria_ep10",
        "voicepack": "df_victoria",
        "lang": "d",
    },
    "martin": {
        "label": "Martin – Deutsch, männlich",
        "backbone": "kikiri_german_martin_ep10",
        "voicepack": "dm_martin",
        "lang": "d",
    },
    "heart": {
        "label": "Heart – Englisch, weiblich",
        "backbone": "kokoro-v1_0",
        "voicepack": "af_heart",
        "lang": "a",
    },
    "michael": {
        "label": "Michael – Englisch, männlich",
        "backbone": "kokoro-v1_0",
        "voicepack": "am_michael",
        "lang": "a",
    },
}

DEFAULT_VOICE = "victoria"


def models_dir() -> Path:
    return Path(os.environ.get("MODELS_DIR", "/models"))


def backbone_path(voice_key: str) -> Path:
    return models_dir() / "backbone" / f"{VOICES[voice_key]['backbone']}.pth"


def voicepack_path(voice_key: str) -> Path:
    return models_dir() / "voices" / f"{VOICES[voice_key]['voicepack']}.pt"


def config_path() -> Path:
    return models_dir() / "backbone" / "config.json"


def is_available(voice_key: str) -> bool:
    """Alle Dateien vorhanden, die diese Stimme braucht?"""
    if voice_key not in VOICES:
        return False
    return (
        backbone_path(voice_key).is_file()
        and voicepack_path(voice_key).is_file()
        and config_path().is_file()
    )


def available_voices() -> list[dict]:
    """Nur Stimmen, deren Dateien tatsächlich da sind (für das Dropdown)."""
    return [
        {"key": key, "label": data["label"]}
        for key, data in VOICES.items()
        if is_available(key)
    ]
