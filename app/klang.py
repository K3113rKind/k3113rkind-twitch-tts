"""Nachbearbeitung der fertigen Sprache: Sprechtempo für Qwen.

Qwen kennt keinen Tempo-Parameter; die Ausgabe wird deshalb nachträglich
tonhöhenneutral gestreckt – über ffmpeg mit Rubber Band (librubberband),
ohne Rubber Band per atempo. Kokoro regelt das Tempo selbst.

(Ein Klang-Regler mit Tonhöhenverschiebung wurde getestet und verworfen:
klang mit Phasen-Vocoder hallig, mit Rubber Band blechern.)
"""

from __future__ import annotations

import logging
import subprocess
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _has_rubberband() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True,
                             text=True, timeout=10).stdout
        found = " rubberband " in out
    except Exception:
        found = False
    log.info("Tempo: %s", "Rubber Band" if found else "atempo (kein Rubber Band im ffmpeg)")
    return found


def apply_tempo(samples: np.ndarray, sr: int, tempo: float = 1.0) -> np.ndarray:
    tempo = max(0.5, min(2.0, float(tempo or 1.0)))
    if abs(tempo - 1.0) <= 0.01 or len(samples) == 0:
        return samples
    filt = (f"rubberband=tempo={tempo:.3f}:pitchq=quality:transients=smooth"
            if _has_rubberband() else f"atempo={tempo:.3f}")
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           "-af", filt, "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1"]
    try:
        res = subprocess.run(cmd, input=samples.astype("<f4").tobytes(), capture_output=True,
                             timeout=60, check=True)
        out = np.frombuffer(res.stdout, dtype="<f4").astype(np.float32)
        return out if len(out) else samples
    except Exception as exc:  # darf die Ausgabe nie verhindern
        log.warning("Tempo nicht anwendbar (%s) – gebe unverändert aus", exc)
        return samples
