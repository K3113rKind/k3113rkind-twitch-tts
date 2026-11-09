"""Sprechtempo für eigene Qwen-Stimmen (Klone).

Qwen-Klone haben keinen Tempo-Parameter und ignorieren Anweisungen (nur
CustomVoice/VoiceDesign verstehen "speak slowly"); eine verlangsamte
Vorlage wirkt laut QwenLM/Qwen3-TTS#290 ebenfalls nicht. Also wird die
fertige Ausgabe bearbeitet – mit Verfahren für Sprache statt für Musik:

  auto     erst die Pausen zwischen Wörtern/Sätzen verlängern bzw. kürzen
           (Laute bleiben unangetastet), den Rest per PSOLA – Standard
  pausen   nur Pausen (kann nicht blechern klingen, wirkt aber nur so weit,
           wie Pausen vorhanden sind)
  psola    nur PSOLA: Praat "Lengthen (overlap-add)", pitch-synchron –
           ganze Stimmschwingungen werden verdoppelt/entfernt, Stimmfarbe und
           Tonhöhe bleiben, kein Phasen-Vocoder
  atempo   ffmpeg atempo (bisheriges Verfahren)
"""

from __future__ import annotations

import logging
import subprocess

import numpy as np

log = logging.getLogger(__name__)

METHODS = ("auto", "pausen", "psola", "atempo")

FRAME_S = 0.01          # Analyse in 10-ms-Schritten
QUIET_DB = 40.0         # leiser als Spitze - 40 dB = Pause
MIN_PAUSE_S = 0.04      # kürzere Lücken sind keine Pausen (Verschlusslaute)
KEEP_PAUSE_S = 0.08     # beim Beschleunigen bleibt jede Pause mindestens so lang
MAX_PAUSE_FACTOR = 2.5  # Pausen höchstens auf das 2,5-Fache dehnen
_warned: set[str] = set()


def _warn_once(key: str, text: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(text, *args)


# ------------------------------------------------------------------ Pausen
def _pauses(samples: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Innere Pausen als (Start, Ende) in Samples – Anfang/Ende zählen nicht."""
    frame = max(1, int(FRAME_S * sr))
    n = len(samples) // frame
    if n < 3:
        return []
    rms = np.sqrt(np.mean(samples[: n * frame].reshape(n, frame) ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-9)
    quiet = db < db.max() - QUIET_DB
    loud = np.nonzero(~quiet)[0]
    if len(loud) == 0:
        return []
    first, last = loud[0], loud[-1]
    out, start = [], None
    for i in range(first, last + 1):
        if quiet[i] and start is None:
            start = i
        elif not quiet[i] and start is not None:
            if (i - start) * FRAME_S >= MIN_PAUSE_S:
                out.append((start * frame, i * frame))
            start = None
    return out


def _scale_pauses(samples: np.ndarray, sr: int, pauses, target_change: int) -> np.ndarray:
    """Pausen anteilig um insgesamt target_change Samples verlängern (positiv)
    oder kürzen (negativ). Eingefügt bzw. entfernt wird in der Mitte jeder
    Pause; die Pause ist dort ohnehin fast still."""
    total = sum(e - s for s, e in pauses)
    if not pauses or total == 0 or target_change == 0:
        return samples
    keep = int(KEEP_PAUSE_S * sr)
    parts, pos = [], 0
    for s, e in pauses:
        mid = (s + e) // 2
        change = int(round(target_change * (e - s) / total))
        if change > 0:
            parts += [samples[pos:mid], np.zeros(change, dtype=np.float32)]
            pos = mid
        elif change < 0:
            cut = min(-change, max(0, (e - s) - keep))
            a = mid - cut // 2
            parts.append(samples[pos:a])
            pos = a + cut
    parts.append(samples[pos:])
    return np.concatenate(parts).astype(np.float32)


# ------------------------------------------------------------------- PSOLA
def _psola(samples: np.ndarray, sr: int, factor: float) -> np.ndarray | None:
    """Dauer mit Praat "Lengthen (overlap-add)" ändern (factor > 1 = länger).
    None, wenn Praat (parselmouth) fehlt."""
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        _warn_once("psola", "Tempo: parselmouth fehlt – nehme atempo statt PSOLA")
        return None
    factor = max(0.34, min(2.9, factor))  # Praat: Faktor > 3 geht nicht
    snd = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=float(sr))
    out = call(snd, "Lengthen (overlap-add)", 75, 600, factor)
    res = np.nan_to_num(np.asarray(out.values, dtype=np.float32).reshape(-1))
    # Plausibilität: Länge ungefähr wie gewünscht, Pegel nicht verschwunden –
    # sonst lieber unverändert als stumm/kaputt ausgeben
    rms_in = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
    rms_out = float(np.sqrt(np.mean(res ** 2))) if len(res) else 0.0
    want = len(samples) * factor
    if not len(res) or abs(len(res) - want) > 0.2 * want or (rms_in > 0 and rms_out < 0.2 * rms_in):
        _warn_once("psola-check", "Tempo: PSOLA-Ergebnis unplausibel (Länge %d statt ~%d, Pegel %.4f "
                   "statt %.4f) – Abschnitt ohne PSOLA", len(res), int(want), rms_out, rms_in)
        return samples
    return np.clip(res, -1.0, 1.0)


def _atempo(samples: np.ndarray, sr: int, tempo: float) -> np.ndarray:
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           "-af", f"atempo={tempo:.3f}", "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1"]
    res = subprocess.run(cmd, input=samples.astype("<f4").tobytes(), capture_output=True,
                         timeout=60, check=True)
    out = np.frombuffer(res.stdout, dtype="<f4").astype(np.float32)
    return out if len(out) else samples


def _duration(samples: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Restliche Dauer-Änderung: PSOLA, sonst atempo."""
    if abs(factor - 1.0) <= 0.005:
        return samples
    out = _psola(samples, sr, factor)
    return out if out is not None else _atempo(samples, sr, 1.0 / factor)


# --------------------------------------------------------------- Einstieg
def apply_tempo(samples: np.ndarray, sr: int, tempo: float = 1.0,
                method: str = "auto") -> np.ndarray:
    """tempo < 1 = langsamer (0,9 = 90 %), > 1 = schneller."""
    tempo = max(0.5, min(2.0, float(tempo or 1.0)))
    if abs(tempo - 1.0) <= 0.01 or len(samples) == 0:
        return samples
    samples = samples.astype(np.float32)
    try:
        if method == "atempo":
            return _atempo(samples, sr, tempo)
        if method == "psola":
            return _duration(samples, sr, 1.0 / tempo)
        # Pausen zuerst
        want = int(round(len(samples) / tempo)) - len(samples)  # + länger, - kürzer
        pauses = _pauses(samples, sr)
        total = sum(e - s for s, e in pauses)
        if want > 0:
            possible = int(total * (MAX_PAUSE_FACTOR - 1))
        else:
            possible = -sum(max(0, (e - s) - int(KEEP_PAUSE_S * sr)) for s, e in pauses)
        by_pauses = max(want, possible) if want < 0 else min(want, possible)
        out = _scale_pauses(samples, sr, pauses, by_pauses)
        rest = (len(samples) + want) / max(1, len(out))
        log.info("Tempo %.2f (%s): Pausen %+.2f s (%d Pausen), Rest per PSOLA ×%.3f",
                 tempo, method, by_pauses / sr, len(pauses), 1.0 if method == "pausen" else rest)
        if method == "pausen":
            return out
        return _duration(out, sr, rest)
    except Exception as exc:  # darf die Ausgabe nie verhindern
        log.warning("Tempo nicht anwendbar (%s) – gebe unverändert aus", exc)
        return samples


# --------------------------------------------------- Lautheitsangleichung
LOUDNESS_TARGET_DB = -20.0   # Ziel-Pegel der Sprache (RMS über die gesprochenen Teile)
LOUDNESS_MAX_GAIN_DB = 12.0  # höchstens so viel lauter/leiser machen


def normalize_loudness(samples: np.ndarray, sr: int, target_db: float = LOUDNESS_TARGET_DB) -> np.ndarray:
    """Sprache auf einen einheitlichen Pegel bringen – damit Kokoro, Pocket
    und Qwen (und verschiedene Stimmen) gleich laut klingen. Gemessen wird nur
    über die gesprochenen Teile (Pausen zählen nicht), Spitzen werden auf
    -1 dBFS begrenzt."""
    if len(samples) == 0:
        return samples
    x = samples.astype(np.float32)
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
    gain_db = max(-LOUDNESS_MAX_GAIN_DB, min(LOUDNESS_MAX_GAIN_DB, target_db - level))
    y = x * (10 ** (gain_db / 20))
    peak = float(np.max(np.abs(y)))
    if peak > 0.89:
        y *= 0.89 / peak
    return y.astype(np.float32)
