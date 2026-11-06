"""Nachbearbeitung der fertigen Sprache: Tempo und Klang (hell <-> warm).

Läuft über ffmpeg mit Rubber Band (librubberband) – hochwertiges
Time-Stretching und Pitch-Shifting, das bei Sprache deutlich sauberer klingt
als ein einfacher Phasen-Vocoder (der erzeugte Echo/Hall).

  tempo   1.0 = unverändert, 0.6 = langsamer (Tonhöhe bleibt). Wird für Qwen
          genutzt, das selbst kein Sprechtempo kennt; Kokoro regelt das selbst.
  warmth  -1 hell/frisch … 0 neutral … +1 warm/tief:
          Tonhöhe bis ±2 Halbtöne bei erhaltenen Formanten (klingt nach
          derselben Person, nur tiefer/höher), dazu sanfte Tiefen/Höhen.

Fehlt Rubber Band im ffmpeg, fällt das Tempo auf ffmpegs atempo zurück und
die Tonhöhe bleibt unverändert.
"""

from __future__ import annotations

import logging
import subprocess
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)

MAX_SEMITONES = 2.0


@lru_cache(maxsize=1)
def _has_rubberband() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True,
                             text=True, timeout=10).stdout
        found = " rubberband " in out
    except Exception:
        found = False
    log.info("Klang/Tempo: %s", "Rubber Band verfügbar" if found
             else "kein Rubber Band im ffmpeg – Tempo über atempo, Tonhöhe bleibt unverändert")
    return found


def _filters(sr: int, warmth: float, tempo: float) -> list[str]:
    chain = []
    pitch = 2 ** (-MAX_SEMITONES * warmth / 12)
    if _has_rubberband():
        if abs(tempo - 1) > 0.01 or abs(pitch - 1) > 0.001:
            chain.append(f"rubberband=tempo={tempo:.3f}:pitch={pitch:.4f}"
                         ":formant=preserved:pitchq=quality:transients=smooth")
    elif abs(tempo - 1) > 0.01:
        chain.append(f"atempo={max(0.5, min(2.0, tempo)):.3f}")
    if abs(warmth) >= 0.02:
        low = 4.0 * warmth if warmth > 0 else 2.5 * warmth
        high = -3.0 * warmth if warmth > 0 else -2.5 * warmth
        chain.append(f"bass=g={low:.2f}:f=250:w=0.7,treble=g={high:.2f}:f=6000:w=0.7")
    if chain:
        chain.append("alimiter=limit=0.95:level=disabled")
    return chain


def apply(samples: np.ndarray, sr: int, warmth: float = 0.0, tempo: float = 1.0) -> np.ndarray:
    warmth = max(-1.0, min(1.0, float(warmth or 0.0)))
    tempo = max(0.5, min(2.0, float(tempo or 1.0)))
    chain = _filters(sr, warmth, tempo)
    if not chain or len(samples) == 0:
        return samples
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           "-af", ",".join(chain), "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1"]
    try:
        res = subprocess.run(cmd, input=samples.astype("<f4").tobytes(), capture_output=True,
                             timeout=60, check=True)
        out = np.frombuffer(res.stdout, dtype="<f4").astype(np.float32)
        return out if len(out) else samples
    except Exception as exc:  # Nachbearbeitung darf die Ausgabe nie verhindern
        log.warning("Klang/Tempo nicht anwendbar (%s) – gebe unbearbeitet aus", exc)
        return samples
