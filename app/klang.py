"""Sprechtempo für eigene Qwen-Stimmen (Klone).

Qwen-Klone kennen keinen Tempo-Parameter und ignorieren Anweisungen – sie
sprechen im Tempo ihrer Referenzaufnahme. Deshalb wird die fertige Ausgabe
tonhöhenneutral gestreckt. Standard ist ffmpeg "atempo" (Zeitbereich,
überlappende Abschnitte): bei Sprache und kleinen Änderungen (0,8–1,25×) am
natürlichsten, kein Hall/Blech wie bei Phasen-Vocodern. Rubber Band (vorher
verwendet, klang verfremdet) ist per QWEN_TEMPO=rubberband wählbar.
Preset-Stimmen bekommen das Tempo weiter als Anweisung (voices.speed_instruct).
"""

from __future__ import annotations

import logging
import os
import subprocess

import numpy as np

log = logging.getLogger(__name__)

TEMPO_ENGINE = os.environ.get("QWEN_TEMPO", "atempo").strip().lower()


def apply_tempo(samples: np.ndarray, sr: int, tempo: float = 1.0) -> np.ndarray:
    tempo = max(0.5, min(2.0, float(tempo or 1.0)))
    if abs(tempo - 1.0) <= 0.01 or len(samples) == 0:
        return samples
    filt = (f"rubberband=tempo={tempo:.3f}:pitchq=quality"
            if TEMPO_ENGINE == "rubberband" else f"atempo={tempo:.3f}")
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
