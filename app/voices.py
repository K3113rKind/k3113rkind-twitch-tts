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

import importlib.util
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

# --- Qwen3-TTS (Test) -------------------------------------------------------
# Preset-Stimmen der CustomVoice-Modelle, deutsch gesprochen. Auf CPU deutlich
# langsamer als Echtzeit (Benchmark R740, 16 Threads: RTF ~2,9 bei 0.6B,
# ~4,0 bei 1.7B) – für Testtexte gedacht, nicht für einen lebhaften Chat.
# Die Modelle werden nicht heruntergeladen, sondern aus einem gemounteten
# Hugging-Face-Cache gelesen (HF_HOME, siehe docker-compose.yml).
QWEN_MODELS = {
    "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}
QWEN_SPEAKERS = [
    ("serena", "Serena", "weiblich"),
    ("vivian", "Vivian", "weiblich"),
    ("ryan", "Ryan", "männlich"),
    ("aiden", "Aiden", "männlich"),
]
for _size, _repo in QWEN_MODELS.items():
    for _key, _name, _gender in QWEN_SPEAKERS:
        VOICES[f"qwen-{_size}-{_key}"] = {
            "label": f"Qwen {_size.upper()} · {_name} – Deutsch, {_gender} (Test, langsam)",
            "engine": "qwen",
            "model": _repo,
            "speaker": _name,
            "lang": "d",  # für "sagt"/"says" bei der Namensansage
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


def engine(voice_key: str) -> str:
    return VOICES[voice_key].get("engine", "kokoro")


def hf_snapshot(repo_id: str) -> Path | None:
    """Lokaler Snapshot eines Hugging-Face-Modells im Cache, falls vorhanden."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snaps = hub / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in sorted(snaps.iterdir()):
        if (snap / "config.json").is_file():
            return snap
    return None


def is_available(voice_key: str) -> bool:
    """Alle Dateien vorhanden, die diese Stimme braucht?"""
    if voice_key not in VOICES:
        return False
    if engine(voice_key) == "qwen":
        return (
            importlib.util.find_spec("qwen_tts") is not None
            and hf_snapshot(VOICES[voice_key]["model"]) is not None
        )
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
