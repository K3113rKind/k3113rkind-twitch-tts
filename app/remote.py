"""Freigegebene Stimmen der Administrativ-Instanz (Test-CT).

Die Administrativ-Instanz entscheidet über "Stimmen verwalten", welche
Stimmen hier erscheinen, und überträgt die Liste per /api/remote/sync
(geschützt mit remote_key, den diese Instanz beim ersten Start erzeugt).
In der Auswahl stehen sie wie jede andere Stimme – ohne eigene Gruppe,
ohne Einstellungen in der GUI.

Erzeugt wird der Ton dort (/api/share/synthesize, gleicher Schlüssel). Ist
die Administrativ-Instanz nicht erreichbar, liest die Standardstimme vor
(siehe tts.py).

Schlüssel in der Stimmauswahl: "remote-<Schlüssel der Administrativ-Instanz>".
"""

from __future__ import annotations

import io
import logging
import wave

log = logging.getLogger(__name__)

PREFIX = "remote-"
CONNECT_TIMEOUT = 5.0
# Langsame Engines (Qwen auf CPU) brauchen für längere Texte eine Weile
SYNTH_TIMEOUT = 180.0

REGISTRY: "RemoteVoices | None" = None  # wird in main.py gesetzt


def is_remote(voice_key: str) -> bool:
    return str(voice_key).startswith(PREFIX)


def clean_voices(raw) -> list[dict]:
    """Übertragene Liste prüfen: nur Schlüssel, Name und Sprache."""
    out, seen = [], set()
    for v in raw if isinstance(raw, list) else []:
        if not isinstance(v, dict):
            continue
        key = str(v.get("key") or "").strip()[:120]
        label = str(v.get("label") or "").strip()[:80]
        if not key or not label or key in seen:
            continue
        seen.add(key)
        out.append({"key": key, "label": label, "lang": str(v.get("lang") or "d")[:4]})
    return out


class RemoteVoices:
    def __init__(self, config) -> None:
        self.config = config

    def _by_key(self) -> dict[str, dict]:
        return {PREFIX + v["key"]: v for v in self.config.get("remote_voices") or []}

    def voices(self) -> list[dict]:
        return [{"key": k, "label": v["label"]} for k, v in self._by_key().items()]

    def get(self, voice_key: str) -> dict | None:
        return self._by_key().get(voice_key)

    def lang(self, voice_key: str) -> str:
        spec = self.get(voice_key)
        return spec["lang"] if spec else "d"

    def synthesize(self, text: str, voice_key: str, speed: float = 1.0):
        """Liefert (samples float32, sample_rate). Wirft bei jedem Fehler."""
        import httpx
        import numpy as np

        spec = self.get(voice_key)
        url = (self.config.get("remote_url") or "").rstrip("/")
        if spec is None or not url:
            raise RuntimeError("Stimme nicht (mehr) freigegeben")
        r = httpx.post(f"{url}/api/share/synthesize",
                       headers={"Authorization": f"Bearer {self.config.get('remote_key')}"},
                       json={"voice": spec["key"], "text": text, "speed": speed},
                       timeout=httpx.Timeout(SYNTH_TIMEOUT, connect=CONNECT_TIMEOUT))
        if r.status_code != 200:
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = r.text[:200]
            raise RuntimeError(f"Administrativ-Instanz: {detail or r.status_code}")
        with wave.open(io.BytesIO(r.content), "rb") as w:
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return pcm.astype(np.float32) / 32768.0, sr
