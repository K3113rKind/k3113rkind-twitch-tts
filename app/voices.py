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
# "group" = Abschnitt in "Stimmen verwalten"; "default": False = Katalogstimme,
# die erst nach dem Aktivieren in der Auswahl erscheint.
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
# Alle Preset-Sprecher der CustomVoice-Modelle (Name, Geschlecht, Muttersprache,
# standardmäßig sichtbar). Sie sprechen alle Deutsch, ggf. mit Akzent.
QWEN_SPEAKERS = [
    ("serena", "Serena", "weiblich", "Chinesisch", True),
    ("vivian", "Vivian", "weiblich", "Chinesisch", True),
    ("ryan", "Ryan", "männlich", "Englisch", True),
    ("aiden", "Aiden", "männlich", "Englisch", True),
    ("ono_anna", "Ono_Anna", "weiblich", "Japanisch", False),
    ("sohee", "Sohee", "weiblich", "Koreanisch", False),
    ("uncle_fu", "Uncle_Fu", "männlich", "Chinesisch", False),
    ("dylan", "Dylan", "männlich", "Chinesisch (Peking)", False),
    ("eric", "Eric", "männlich", "Chinesisch (Sichuan)", False),
]
for _size, _repo in QWEN_MODELS.items():
    for _key, _name, _gender, _native, _default in QWEN_SPEAKERS:
        VOICES[f"qwen-{_size}-{_key}"] = {
            "label": f"Qwen {_size.upper()} · {_name.replace('_', ' ')} – Deutsch, {_gender}"
                     + ("" if _default else f" (Vorlage {_native})") + " (Test, langsam)",
            "engine": "qwen",
            "model": _repo,
            "speaker": _name,
            "lang": "d",  # für "sagt"/"says" bei der Namensansage
            "group": f"Qwen {_size.upper()}",
            "default": _default,
        }

# --- Eigene Stimmen (Qwen-Stimmklonen, Test) ---------------------------------
# Jedes Paar NAME.wav + NAME.txt im Referenzordner (Mount /ref, befüllt von
# tools/stimme-klonen.sh) wird zu je einer Stimme pro Base-Modell. Der Ordner
# wird bei jedem Abruf neu gelesen – neue Stimmen erscheinen ohne Neustart.
QWEN_BASE_MODELS = {
    "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
CLONE_ENGINE = "qwen-clone"

# --- Pocket TTS (Kyutai, Test) -----------------------------------------------
# Kleines CPU-Modell (100M). "german" = schnell, "german_24l" = 24 Schichten,
# bessere Qualität, langsamer. Gewichte lädt app/pocket.py beim ersten Gebrauch.
POCKET_ENGINE = "pocket"
POCKET_CLONE_ENGINE = "pocket-clone"
POCKET_MODELS = {
    "pocket": ("german", "Pocket"),
    "pocket24": ("german_24l", "Pocket 24L"),
}
# Weibliche Katalogstimmen von Kyutai. Es gibt sie nur als englische (bzw.
# französische/spanische) Aufnahmen – sie werden daraus geklont und sprechen
# Deutsch, ggf. mit leichtem Akzent. Braucht das Modell mit Stimmklonen (HF_TOKEN).
POCKET_FEMALE = [
    ("alba", "Alba", "Englisch"),
    ("anna", "Anna", "Englisch"),
    ("azelma", "Azelma", "Englisch"),
    ("caro_davy", "Caro", "Englisch"),
    ("cosette", "Cosette", "Englisch"),
    ("eponine", "Eponine", "Englisch"),
    ("eve", "Eve", "Englisch"),
    ("fantine", "Fantine", "Englisch"),
    ("jane", "Jane", "Englisch"),
    ("mary", "Mary", "Englisch"),
    ("vera", "Vera", "Englisch"),
    ("estelle", "Estelle", "Französisch"),
    ("lola", "Lola", "Spanisch"),
]
for _key, (_lang, _name) in POCKET_MODELS.items():
    VOICES[f"{_key}-juergen"] = {
        "label": f"{_name} · Jürgen – Deutsch, männlich (Test)",
        "engine": POCKET_ENGINE,
        "language": _lang,
        "voice": "juergen",
        "lang": "d",
        "group": _name,
    }
    for _vkey, _vname, _origin in POCKET_FEMALE:
        VOICES[f"{_key}-{_vkey}"] = {
            "label": f"{_name} · {_vname} – Deutsch, weiblich (Vorlage {_origin}, Test)",
            "engine": POCKET_ENGINE,
            "language": _lang,
            "voice": f"audio:{_vkey}",
            "lang": "d",
            "group": _name,
        }

# --- NeuTTS / Chatterbox (Test) ---------------------------------------------
# Laufen in eigenen Python-Umgebungen (/opt/neutts, /opt/chatterbox) über
# app/ext_pool.py – deren Abhängigkeiten vertragen sich nicht mit dem Rest.
NEUTTS_ENGINE = "neutts"
CHATTERBOX_ENGINE = "chatterbox"
EXT_PYTHON = {
    NEUTTS_ENGINE: os.environ.get("NEUTTS_PYTHON", "/opt/neutts/bin/python"),
    CHATTERBOX_ENGINE: os.environ.get("CHATTERBOX_PYTHON", "/opt/chatterbox/bin/python"),
}
EXT_ENGINES = tuple(EXT_PYTHON)
NEUTTS_SAMPLES = Path(os.environ.get("NEUTTS_SAMPLES", "/opt/neutts/samples"))

VOICES["neutts-greta"] = {
    "label": "NeuTTS · Greta – Deutsch, weiblich (Test)",
    "engine": NEUTTS_ENGINE,
    "ref_codes": str(NEUTTS_SAMPLES / "greta.pt"),
    "ref_txt": str(NEUTTS_SAMPLES / "greta.txt"),
    "lang": "d",
    "group": "NeuTTS",
}
VOICES["chatterbox-standard"] = {
    "label": "Chatterbox · Standardstimme – Deutsch (Test, langsam)",
    "engine": CHATTERBOX_ENGINE,
    "lang": "d",
    "group": "Chatterbox",
}

# Eigene Kokoro-Stimmpakete (app/kokoro_walk.py): /ref/kokoro/NAME.pt, laufen
# auf dem deutschen Victoria-Backbone. Werden während der Suche laufend
# verbessert – die Datei wird dabei ersetzt.
KOKORO_CUSTOM_BACKBONE = "kikiri_german_victoria_ep10"

# --- Kokoro: englische Originalstimmen (Katalog) ------------------------------
# Laufen auf dem offiziellen Kokoro-82M (wie Heart/Michael). Das Stimmpaket wird
# beim Aktivieren einmalig aus hexgrad/Kokoro-82M in den HF-Cache (/hf) geladen,
# weil /models schreibgeschützt eingebunden ist.
KOKORO_STOCK_REPO = "hexgrad/Kokoro-82M"
KOKORO_STOCK = {
    "a": ("Englisch (US)", {
        "f": ["alloy", "aoede", "bella", "jessica", "kore", "nicole", "nova", "river", "sarah", "sky"],
        "m": ["adam", "echo", "eric", "fenrir", "liam", "onyx", "puck", "santa"],
    }),
    "b": ("Englisch (UK)", {
        "f": ["alice", "emma", "isabella", "lily"],
        "m": ["daniel", "fable", "george", "lewis"],
    }),
}
for _key in ("victoria", "martin", "heart", "michael"):
    VOICES[_key]["group"] = "Kokoro"
for _lang, (_lname, _by_gender) in KOKORO_STOCK.items():
    for _g, _names in _by_gender.items():
        for _n in _names:
            VOICES[f"kokoro-{_lang}{_g}_{_n}"] = {
                "label": f"{_n.capitalize()} – {_lname}, {'weiblich' if _g == 'f' else 'männlich'}",
                "backbone": "kokoro-v1_0",
                "voicepack": f"{_lang}{_g}_{_n}",
                "hf_voicepack": True,
                "lang": _lang,
                "group": "Kokoro",
                "default": False,
            }


def kokoro_stock_path(pack: str) -> Path | None:
    """Heruntergeladenes Original-Stimmpaket im HF-Cache, falls vorhanden."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snaps = hub / f"models--{KOKORO_STOCK_REPO.replace('/', '--')}" / "snapshots"
    if snaps.is_dir():
        for snap in sorted(snaps.iterdir(), reverse=True):
            f = snap / "voices" / f"{pack}.pt"
            if f.is_file():
                return f
    return None


def needs_download(voice_key: str) -> bool:
    spec = VOICES.get(voice_key, {})
    return bool(spec.get("hf_voicepack")) and kokoro_stock_path(spec["voicepack"]) is None


# --- Pocket: weitere Katalogstimmen von Kyutai --------------------------------
# Die vollständige Liste kennt erst das installierte Paket pocket-tts. Sie wird
# beim ersten Bedarf eingelesen; alles außer den Standardstimmen ist versteckt.
_pocket_catalog_done = False


def ensure_pocket_catalog() -> None:
    global _pocket_catalog_done
    if _pocket_catalog_done or importlib.util.find_spec("pocket_tts") is None:
        return
    _pocket_catalog_done = True
    try:
        import torch

        before = torch.get_num_threads()
        from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

        torch.set_num_threads(before)  # pocket_tts setzt beim Import 1 Thread
    except Exception:
        return
    for name in sorted(_ORIGINS_OF_PREDEFINED_VOICES):
        for key, (lang, model_name) in POCKET_MODELS.items():
            vkey = f"{key}-{name}"
            if vkey in VOICES or name == "juergen":
                continue
            VOICES[vkey] = {
                "label": f"{model_name} · {name.replace('_', ' ').title()} – Deutsch (Katalogstimme, Test)",
                "engine": POCKET_ENGINE,
                "language": lang,
                "voice": f"audio:{name}",
                "lang": "d",
                "group": model_name,
                "default": False,
            }


def is_visible(voice_key: str, visibility: dict | None = None) -> bool:
    """Sichtbar in der Stimmauswahl? Eigene Einstellung vor Standardwert."""
    if visibility and voice_key in visibility:
        return bool(visibility[voice_key])
    return bool(VOICES.get(voice_key, {}).get("default", True))


# Schlüssel-Präfixe einer eigenen Stimme NAME: klon-<präfix>-NAME
CLONE_PREFIXES = [*QWEN_BASE_MODELS, *POCKET_MODELS, "kokoro", *EXT_ENGINES]


def clone_keys(name: str) -> set[str]:
    return {f"klon-{prefix}-{name}" for prefix in CLONE_PREFIXES}


def ref_dir() -> Path:
    return Path(os.environ.get("QWEN_REF_PATH", "/ref"))


def sync_clone_voices() -> None:
    """Eigene Stimmen mit dem Referenzordner abgleichen (VOICES in place)."""
    wanted: dict[str, dict] = {}
    folder = ref_dir()
    if folder.is_dir():
        for wav in sorted(folder.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if not txt.is_file():
                continue
            for size, repo in QWEN_BASE_MODELS.items():
                wanted[f"klon-{size}-{wav.stem}"] = {
                    "label": f"{wav.stem} · Qwen {size.upper()} – Deutsch (Test, langsam)",
                    "engine": CLONE_ENGINE,
                    "model": repo,
                    "ref_wav": str(wav),
                    "ref_txt": str(txt),
                    "lang": "d",
                    "group": "Eigene Stimmen",
                }
            for key, (lang, model_name) in POCKET_MODELS.items():
                wanted[f"klon-{key}-{wav.stem}"] = {
                    "label": f"{wav.stem} · {model_name} – Deutsch (Test)",
                    "engine": POCKET_CLONE_ENGINE,
                    "language": lang,
                    "ref_wav": str(wav),
                    "lang": "d",
                    "group": "Eigene Stimmen",
                }
            wanted[f"klon-neutts-{wav.stem}"] = {
                "label": f"{wav.stem} · NeuTTS – Deutsch (Test)",
                "engine": NEUTTS_ENGINE,
                "ref_wav": str(wav),
                "ref_txt": str(txt),
                "lang": "d",
                "group": "Eigene Stimmen",
                "custom_ext": True,
            }
            wanted[f"klon-chatterbox-{wav.stem}"] = {
                "label": f"{wav.stem} · Chatterbox – Deutsch (Test, langsam)",
                "engine": CHATTERBOX_ENGINE,
                "ref_wav": str(wav),
                "lang": "d",
                "group": "Eigene Stimmen",
                "custom_ext": True,
            }
        for pack in sorted((folder / "kokoro").glob("*.pt")):
            wanted[f"klon-kokoro-{pack.stem}"] = {
                "label": f"{pack.stem} · Kokoro – Deutsch (Test)",
                "backbone": KOKORO_CUSTOM_BACKBONE,
                "voicepack_file": str(pack),
                "custom_kokoro": True,
                "lang": "d",
                "group": "Eigene Stimmen",
            }
    for key in [k for k, v in list(VOICES.items())
                if v.get("engine") in (CLONE_ENGINE, POCKET_CLONE_ENGINE) or v.get("custom_kokoro")
                or v.get("custom_ext")]:
        if key not in wanted:
            VOICES.pop(key, None)
    VOICES.update(wanted)


# Sprechstile für die Preset-Stimmen (CustomVoice-Modelle verstehen eine
# Regieanweisung). Eigene Stimmen (Base-Modelle) können das nicht – dort
# kommt der Stil allein aus der Vorlage.
QWEN_STYLES = {
    "neutral": None,
    "froehlich": "Speak in a cheerful, upbeat and friendly tone.",
    "aufgeregt": "Speak with high energy and excitement, lively and a bit faster.",
    "ernst": "Speak in a serious, calm and matter-of-fact tone.",
    "gelangweilt": "Speak in a bored, flat, monotone and unenthusiastic tone.",
    "ruhig": "Speak slowly and calmly with a warm, soft and gentle voice.",
    "verfuehrerisch": "Speak in a low, slow, sultry and seductive voice, slightly breathy and intimate.",
    "fluesternd": "Whisper softly and intimately, very quiet and close.",
    "streng": "Speak in a strict, cold and commanding tone.",
}



def speed_instruct(instruct: str | None, speed: float) -> str | None:
    """Sprechtempo als Anweisung an die Qwen-Preset-Stimmen anhängen.
    Wirkt nur bei CustomVoice (Klone ignorieren instruct); Kokoro nutzt speed direkt."""
    speed = float(speed or 1.0)
    if speed <= 0.7:
        hint = "Speak very slowly, with a relaxed pace and clear pauses."
    elif speed <= 0.9:
        hint = "Speak a bit slower than usual, at a calm pace."
    elif speed >= 1.4:
        hint = "Speak very fast."
    elif speed >= 1.1:
        hint = "Speak a bit faster than usual."
    else:
        return instruct
    return f"{instruct} {hint}" if instruct else hint


DEFAULT_VOICE = "victoria"


def models_dir() -> Path:
    return Path(os.environ.get("MODELS_DIR", "/models"))


def backbone_path(voice_key: str) -> Path:
    return models_dir() / "backbone" / f"{VOICES[voice_key]['backbone']}.pth"


def voicepack_path(voice_key: str) -> Path:
    spec = VOICES[voice_key]
    if spec.get("voicepack_file"):
        return Path(spec["voicepack_file"])
    if spec.get("hf_voicepack"):
        return kokoro_stock_path(spec["voicepack"]) or Path("/nonexistent") / f"{spec['voicepack']}.pt"
    return models_dir() / "voices" / f"{spec['voicepack']}.pt"


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


def is_qwen(voice_key: str) -> bool:
    return engine(voice_key) in ("qwen", CLONE_ENGINE)


def is_pocket(voice_key: str) -> bool:
    return engine(voice_key) in (POCKET_ENGINE, POCKET_CLONE_ENGINE)


def is_ext(voice_key: str) -> bool:
    """NeuTTS / Chatterbox (eigene Python-Umgebung)?"""
    return engine(voice_key) in EXT_ENGINES


def is_chunked(voice_key: str) -> bool:
    """Lange Texte abschnittsweise erzeugen und schon während der Berechnung spielen."""
    return is_qwen(voice_key) or is_pocket(voice_key) or is_ext(voice_key)


def chunk_chars(voice_key: str) -> int | None:
    """Höchstlänge eines Abschnitts (None = Standard). NeuTTS hat ein kurzes
    Kontextfenster (2048 Token inkl. Referenz) – lieber kleinere Stücke."""
    return 180 if engine(voice_key) == NEUTTS_ENGINE else None


def is_available(voice_key: str) -> bool:
    """Alle Dateien vorhanden, die diese Stimme braucht?"""
    if voice_key.startswith("klon-"):
        sync_clone_voices()
    if voice_key not in VOICES:
        return False
    if is_ext(voice_key):
        spec = VOICES[voice_key]
        refs = [spec[k] for k in ("ref_wav", "ref_txt", "ref_codes") if k in spec]
        return Path(EXT_PYTHON[spec["engine"]]).is_file() and all(Path(r).is_file() for r in refs)
    if is_pocket(voice_key):
        spec = VOICES[voice_key]
        return (importlib.util.find_spec("pocket_tts") is not None
                and ("ref_wav" not in spec or Path(spec["ref_wav"]).is_file()))
    if is_qwen(voice_key):
        spec = VOICES[voice_key]
        return (
            importlib.util.find_spec("qwen_tts") is not None
            and hf_snapshot(spec["model"]) is not None
            and (engine(voice_key) == "qwen"
                 or (Path(spec["ref_wav"]).is_file() and Path(spec["ref_txt"]).is_file()))
        )
    return (
        backbone_path(voice_key).is_file()
        and voicepack_path(voice_key).is_file()
        and config_path().is_file()
    )


def display_name(voice_key: str, custom: dict | None = None) -> str:
    """Anzeigename: eigener Name aus der Config, sonst der Standardname."""
    if custom and custom.get(voice_key):
        return custom[voice_key]
    return VOICES[voice_key]["label"]


def available_voices(custom: dict | None = None, visibility: dict | None = None) -> list[dict]:
    """Nur sichtbare Stimmen, deren Dateien tatsächlich da sind (für das Dropdown)."""
    sync_clone_voices()
    ensure_pocket_catalog()
    return [
        {"key": key, "label": display_name(key, custom), "default_label": data["label"],
         "qwen": is_qwen(key), "styleable": data.get("engine") == "qwen",
         "pocket": is_pocket(key), "custom": data.get("group") == "Eigene Stimmen"}
        for key, data in list(VOICES.items())
        if is_visible(key, visibility) and is_available(key)
    ]


def voice_catalog(custom: dict | None = None, visibility: dict | None = None,
                  shared: list | None = None) -> list[dict]:
    """Alle bekannten Stimmen nach Modell gruppiert (für "Stimmen verwalten").
    Kokoro-Katalogstimmen gelten als nutzbar, sobald das Backbone da ist – das
    Stimmpaket wird beim Aktivieren nachgeladen."""
    sync_clone_voices()
    ensure_pocket_catalog()
    groups: dict[str, list[dict]] = {}
    for key, data in list(VOICES.items()):
        if data.get("hf_voicepack"):
            usable = backbone_path(key).is_file() and config_path().is_file()
        else:
            usable = is_available(key)
        if not usable:
            continue
        groups.setdefault(data.get("group", "Sonstige"), []).append({
            "key": key, "label": display_name(key, custom),
            "visible": is_visible(key, visibility),
            "download": needs_download(key),
            "shared": key in (shared or ()),
        })
    return [{"group": g, "voices": v} for g, v in groups.items()]
