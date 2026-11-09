"""Deutsche Tonspur aus den vertonten Sätzen bauen und ins Video einsetzen.

Jeder Satz startet an seiner Untertitel-Zeit. Passt er nicht in die Lücke
bis zum nächsten Satz, wird er bis YT_MAX_SPEEDUP schneller gemacht (erst
Pausen kürzen, dann PSOLA – klang.apply_tempo); was dann noch überhängt,
schiebt die folgenden Sätze etwas nach hinten (holt in Sprechpausen wieder auf).

Ergebnis: video_de.mp4 mit
  Tonspur 1: Deutsch (KI) – Standard, optional mit leisem Original darunter
  Tonspur 2: Original
Video wird nur kopiert (kein Neukodieren).
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import numpy as np

from . import klang

log = logging.getLogger(__name__)

SR = 24000
YT_MAX_SPEEDUP = float(os.environ.get("YT_MAX_SPEEDUP", "1.25") or 1.25)
GAP_S = 0.12  # Mindestabstand zwischen zwei Sätzen


def _resample(samples: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR or len(samples) == 0:
        return samples.astype(np.float32)
    x = np.linspace(0, len(samples) - 1, int(round(len(samples) * SR / sr)))
    return np.interp(x, np.arange(len(samples)), samples).astype(np.float32)


def build_track(sentences: list[dict], audios: list[tuple[np.ndarray, int]],
                duration: float | None) -> tuple[np.ndarray, dict]:
    """Sätze auf die Zeitachse legen. Liefert (Spur, Statistik)."""
    parts = []
    cursor = 0.0
    stats = {"sped_up": 0, "max_delay": 0.0}
    for i, (s, (samples, sr)) in enumerate(zip(sentences, audios)):
        samples = _resample(np.asarray(samples, dtype=np.float32), sr)
        if len(samples) == 0:
            continue
        t = float(s["t"])
        nxt = float(sentences[i + 1]["t"]) if i + 1 < len(sentences) else None
        start = max(t, cursor)
        if nxt is not None:
            slot = max(0.3, nxt - start - GAP_S)
            dur = len(samples) / SR
            if dur > slot:
                factor = min(YT_MAX_SPEEDUP, dur / slot)
                if factor > 1.02:
                    samples = klang.apply_tempo(samples, SR, factor, "auto")
                    stats["sped_up"] += 1
        stats["max_delay"] = max(stats["max_delay"], start - t)
        parts.append((start, samples))
        cursor = start + len(samples) / SR + GAP_S
    end = max(cursor, float(duration or 0)) + 0.5
    track = np.zeros(int(end * SR) + 1, dtype=np.float32)
    for start, samples in parts:
        a = int(start * SR)
        track[a:a + len(samples)] += samples[: max(0, len(track) - a)]
    peak = float(np.max(np.abs(track))) if len(track) else 0.0
    if peak > 0:
        track *= 0.89 / peak  # ca. -1 dBFS
    return track, stats


def write_wav(path: Path, track: np.ndarray) -> None:
    import soundfile as sf

    sf.write(path, track, SR, subtype="PCM_16")


def mux(video: Path, dub_wav: Path, out: Path, original_level: float) -> None:
    """video_de.mp4: deutsche Spur (Standard) + Original als zweite Spur."""
    if original_level > 0:
        # Original leise unter die deutsche Spur mischen
        filt = (f"[0:a]volume={original_level:.2f}[o];[1:a]aresample=48000[d];"
                f"[d][o]amix=inputs=2:duration=longest:normalize=0[de]")
        maps = ["-filter_complex", filt, "-map", "0:v:0", "-map", "[de]", "-map", "0:a:0?"]
    else:
        maps = ["-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0?"]
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-i", str(dub_wav), *maps,
           "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
           "-metadata:s:a:0", "language=deu", "-metadata:s:a:0", "title=Deutsch (KI)",
           "-metadata:s:a:0", "handler_name=Deutsch (KI)",
           "-metadata:s:a:1", "title=Original", "-metadata:s:a:1", "handler_name=Original",
           "-disposition:a:0", "default", "-disposition:a:1", "0",
           "-map_metadata", "0", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True, timeout=1800)
