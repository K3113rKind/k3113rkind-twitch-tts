"""Sprechtempo für Qwen: fertige Ausgabe tonhöhenneutral strecken.

Qwen hat keinen Tempo-Parameter. Wählbare Verfahren (config "qwen_tempo"):

  rubberband   Rubber Band wie in der ersten Fassung (pitchq=quality,
               transients=smooth) – Standard
  rubberband_kurz
               Rubber Band mit kurzem Analysefenster: folgt schnellen
               Lautwechseln genauer, kann bei Sprache klarer klingen
  atempo       ffmpeg atempo (Zeitbereich, überlappende Abschnitte) –
               kein Phasen-Vocoder, bei kleinen Änderungen sehr unauffällig
  anweisung    nichts strecken: Preset-Stimmen bekommen das Tempo als
               Anweisung, eigene Stimmen sprechen im Tempo der Referenz

Fehlt Rubber Band im ffmpeg, wird automatisch atempo genommen.
"""

from __future__ import annotations

import logging
import subprocess
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)

METHODS = ("rubberband", "rubberband_kurz", "atempo", "anweisung")


@lru_cache(maxsize=1)
def _has_rubberband() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True,
                             text=True, timeout=10).stdout
        found = " rubberband " in out
    except Exception:
        found = False
    if not found:
        log.warning("Tempo: kein Rubber Band im ffmpeg – nehme atempo")
    return found


def _filter(method: str, tempo: float) -> str:
    if method.startswith("rubberband") and _has_rubberband():
        if method == "rubberband_kurz":
            return f"rubberband=tempo={tempo:.3f}:pitchq=quality:window=short:transients=mixed"
        return f"rubberband=tempo={tempo:.3f}:pitchq=quality:transients=smooth"
    return f"atempo={tempo:.3f}"


def apply_tempo(samples: np.ndarray, sr: int, tempo: float = 1.0,
                method: str = "rubberband") -> np.ndarray:
    tempo = max(0.5, min(2.0, float(tempo or 1.0)))
    if method == "anweisung" or abs(tempo - 1.0) <= 0.01 or len(samples) == 0:
        return samples
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           "-af", _filter(method, tempo), "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1"]
    try:
        res = subprocess.run(cmd, input=samples.astype("<f4").tobytes(), capture_output=True,
                             timeout=60, check=True)
        out = np.frombuffer(res.stdout, dtype="<f4").astype(np.float32)
        return out if len(out) else samples
    except Exception as exc:  # darf die Ausgabe nie verhindern
        log.warning("Tempo nicht anwendbar (%s) – gebe unverändert aus", exc)
        return samples
