"""Entfernte Stimmen: einzelne, freigegebene Stimmen der Test-Instanz.

Die Test-Instanz gibt über "Stimmen verwalten" gezielt Stimmen frei
(/api/share/voices, /api/share/synthesize, geschützt mit einem Schlüssel).
Hier wird nur diese Liste geholt und Text dorthin zum Erzeugen geschickt –
Modelle, Klonen oder Experimente der Test-Instanz sind nicht erreichbar.

Schlüssel in der Stimmauswahl: "remote-<Schlüssel der Test-Instanz>".
Ist die Test-Instanz nicht erreichbar, bleibt die zuletzt bekannte Liste
stehen; erzeugt wird dann mit der eigenen Standardstimme (siehe tts.py).
"""

from __future__ import annotations

import io
import logging
import threading
import time
import wave

log = logging.getLogger(__name__)

PREFIX = "remote-"
LIST_TIMEOUT = 5.0
# Langsame Engines (Qwen auf CPU) brauchen für längere Texte eine Weile
SYNTH_TIMEOUT = 180.0
REFRESH_SECONDS = 300

REGISTRY: "RemoteVoices | None" = None  # wird in main.py gesetzt


def is_remote(voice_key: str) -> bool:
    return str(voice_key).startswith(PREFIX)


class RemoteVoices:
    def __init__(self, config) -> None:
        self.config = config
        self._voices: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.error: str | None = None
        self.last_ok: float | None = None
        self._configured_for: tuple | None = None

    # ---------------------------------------------------------------- Config
    def _settings(self) -> tuple[str, str]:
        url = (self.config.get("remote_url") or "").strip().rstrip("/")
        return url, (self.config.get("remote_token") or "").strip()

    def enabled(self) -> bool:
        url, token = self._settings()
        return bool(url and token)

    def _headers(self, token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    # ---------------------------------------------------------------- Liste
    def refresh(self) -> dict:
        """Liste der freigegebenen Stimmen neu holen (blockierend)."""
        import httpx

        url, token = self._settings()
        if not (url and token):
            with self._lock:
                self._voices = {}
            self.error = None
            self._configured_for = None
            return self.status()
        # Andere Adresse/Schlüssel: alte Liste gilt nicht mehr
        if self._configured_for != (url, token):
            with self._lock:
                self._voices = {}
            self._configured_for = (url, token)
        try:
            r = httpx.get(f"{url}/api/share/voices", headers=self._headers(token),
                          timeout=LIST_TIMEOUT)
            if r.status_code == 401:
                raise RuntimeError("Schlüssel ungültig")
            r.raise_for_status()
            voices = {
                PREFIX + v["key"]: {
                    "remote_key": v["key"],
                    "label": f"{v['label']}",
                    "lang": v.get("lang") or "d",
                }
                for v in r.json().get("voices", [])
            }
            with self._lock:
                self._voices = voices
            self.error = None
            self.last_ok = time.time()
        except Exception as exc:
            self.error = f"Test-Instanz nicht erreichbar: {exc}"
            log.warning("Entfernte Stimmen: %s", self.error)
        return self.status()

    def status(self) -> dict:
        with self._lock:
            count = len(self._voices)
        return {"enabled": self.enabled(), "count": count, "error": self.error,
                "last_ok": self.last_ok}

    def voices(self) -> list[dict]:
        with self._lock:
            return [{"key": k, "label": v["label"], "remote": True}
                    for k, v in self._voices.items()]

    def get(self, voice_key: str) -> dict | None:
        with self._lock:
            return self._voices.get(voice_key)

    def lang(self, voice_key: str) -> str:
        spec = self.get(voice_key)
        return spec["lang"] if spec else "d"

    # ---------------------------------------------------------------- Audio
    def synthesize(self, text: str, voice_key: str, speed: float = 1.0):
        """Liefert (samples float32, sample_rate). Wirft bei jedem Fehler."""
        import httpx
        import numpy as np

        spec = self.get(voice_key)
        if spec is None:
            raise RuntimeError("Entfernte Stimme nicht (mehr) freigegeben")
        url, token = self._settings()
        if not (url and token):
            raise RuntimeError("Test-Instanz nicht eingerichtet")
        r = httpx.post(f"{url}/api/share/synthesize", headers=self._headers(token),
                       json={"voice": spec["remote_key"], "text": text, "speed": speed},
                       timeout=httpx.Timeout(SYNTH_TIMEOUT, connect=LIST_TIMEOUT))
        if r.status_code != 200:
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = r.text[:200]
            raise RuntimeError(f"Test-Instanz: {detail or r.status_code}")
        with wave.open(io.BytesIO(r.content), "rb") as w:
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return pcm.astype(np.float32) / 32768.0, sr
