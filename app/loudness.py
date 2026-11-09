"""Einheitliche Lautstärke: jede Ansage auf denselben Sprachpegel bringen,
damit verschiedene Stimmen gleich laut klingen."""

from __future__ import annotations

import numpy as np

TARGET_DB = -20.0   # Ziel-Pegel der Sprache (RMS über die gesprochenen Teile)
MAX_GAIN_DB = 12.0  # höchstens so viel lauter/leiser machen


def normalize(samples: np.ndarray, sr: int, target_db: float = TARGET_DB) -> np.ndarray:
    if len(samples) == 0:
        return samples
    x = np.asarray(samples, dtype=np.float32)
    frame = max(1, int(0.02 * sr))
    n = len(x) // frame
    if n == 0:
        return x
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-9)
    voiced = rms[db > db.max() - 30.0]
    if len(voiced) == 0:
        return x
    level = 20 * np.log10(float(np.sqrt(np.mean(voiced ** 2))) + 1e-9)
    gain_db = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, target_db - level))
    y = x * (10 ** (gain_db / 20))
    peak = float(np.max(np.abs(y)))
    if peak > 0.89:
        y *= 0.89 / peak
    return y.astype(np.float32)
