"""Klang-Regler: fertige Sprache wärmer/tiefer oder heller/frischer machen.

Reine Nachbearbeitung, funktioniert deshalb mit jeder Stimme (Kokoro, Qwen,
eigene Stimmen). Wert von -1 bis +1:

  +1  warm/tief:   Tonhöhe -2 Halbtöne, Tiefen +4 dB, Höhen -3 dB
   0  unverändert
  -1  hell/frisch: Tonhöhe +2 Halbtöne, Tiefen -3 dB, Höhen +3 dB

Die Tonhöhe ändert sich ohne Tempoänderung (librosa, Phasen-Vocoder); stärker
als ±2 Halbtöne klingt das schnell künstlich, deshalb die Grenze.
"""

from __future__ import annotations

import logging
import math

import numpy as np

log = logging.getLogger(__name__)

MAX_SEMITONES = 2.0
_warned = False


def _shelf(sr: int, freq: float, gain_db: float, high: bool):
    """Biquad-Kuhfilter (RBJ Audio EQ Cookbook, Steilheit S=1)."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * freq / sr
    cos, alpha = math.cos(w0), math.sin(w0) / 2 * math.sqrt(2)
    sq = 2 * math.sqrt(a) * alpha
    if high:
        b = [a * ((a + 1) + (a - 1) * cos + sq), -2 * a * ((a - 1) + (a + 1) * cos),
             a * ((a + 1) + (a - 1) * cos - sq)]
        den = [(a + 1) - (a - 1) * cos + sq, 2 * ((a - 1) - (a + 1) * cos),
               (a + 1) - (a - 1) * cos - sq]
    else:
        b = [a * ((a + 1) - (a - 1) * cos + sq), 2 * a * ((a - 1) - (a + 1) * cos),
             a * ((a + 1) - (a - 1) * cos - sq)]
        den = [(a + 1) + (a - 1) * cos + sq, -2 * ((a - 1) + (a + 1) * cos),
               (a + 1) + (a - 1) * cos - sq]
    return np.array(b) / den[0], np.array(den) / den[0]


def apply(samples: np.ndarray, sr: int, warmth: float) -> np.ndarray:
    global _warned
    warmth = max(-1.0, min(1.0, float(warmth or 0.0)))
    if abs(warmth) < 0.02 or len(samples) == 0:
        return samples
    from scipy.signal import lfilter

    out = samples.astype(np.float32)
    try:
        import librosa

        out = librosa.effects.pitch_shift(out, sr=sr, n_steps=-MAX_SEMITONES * warmth)
    except Exception as exc:  # Klang soll nie die Ausgabe verhindern
        if not _warned:
            log.warning("Tonhöhe nicht änderbar (%s) – nur Klangfarbe wird angepasst", exc)
            _warned = True

    low_db = 4.0 * warmth if warmth > 0 else 3.0 * warmth
    high_db = -3.0 * warmth
    for freq, gain, high in ((250.0, low_db, False), (6000.0, high_db, True)):
        b, a = _shelf(sr, freq, gain, high)
        out = lfilter(b, a, out)

    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.98:  # nach dem Anheben nicht übersteuern
        out = out * (0.98 / peak)
    return out.astype(np.float32)
