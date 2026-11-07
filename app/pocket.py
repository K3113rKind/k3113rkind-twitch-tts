"""Pocket TTS (Kyutai) – kleines CPU-Modell (100M), Deutsch, Stimmklonen.

Code: MIT (github.com/kyutai-labs/pocket-tts). Gewichte über Hugging Face:
  - kyutai/pocket-tts-without-voice-cloning: frei, nur fertige Stimmen
    (Deutsch: "juergen")
  - kyutai/pocket-tts: mit Stimmklonen, "gated" – Bedingungen auf der
    Modellseite akzeptieren und HF_TOKEN setzen. Ohne Token laufen nur die
    fertigen Stimmen.

Das Image läuft mit HF_HUB_OFFLINE=1. Fehlen die Gewichte im Cache (/hf),
lädt sie ein eigener Prozess einmalig herunter (python -m app.pocket).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Torch-Threads für Pocket: das Modell ist klein, viele Threads bremsen eher
# (Kyutai misst mit 1–2 Kernen). Gilt nur während der Pocket-Berechnung.
POCKET_THREADS = int(os.environ.get("POCKET_THREADS", "2") or 2)


def _env_float(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    return float(value) if value else None


# Klang-Stellschrauben (leer = Standard von Kyutai), siehe
# kyutai-labs.github.io/pocket-tts/CLI Commands/generate/:
#   POCKET_STEPS     Schritte des Flow-Kopfs, Standard 1 – mehr = "higher quality", langsamer
#   POCKET_TEMP      Temperatur, Standard 0.3 – höher = ausdrucksstärker, niedriger = gleichmäßiger
#   POCKET_EOS       Satzende-Schwelle, Standard -4.0 – kleiner = endet früher
#   POCKET_CONTINUE  1 = interne Teilstücke eines Texts setzen das vorige fort
#                    (teacher_force_previous_chunk) statt neu von der Vorlage –
#                    gleichmäßigere Stimme bei längeren Texten, etwas langsamer
POCKET_STEPS = int(os.environ.get("POCKET_STEPS", "1") or 1)
POCKET_TEMP = _env_float("POCKET_TEMP")
POCKET_EOS = _env_float("POCKET_EOS")
POCKET_CONTINUE = os.environ.get("POCKET_CONTINUE", "0").strip().lower() in ("1", "true", "yes")
# Nachlauf nach dem Satzende in Frames à 80 ms (siehe _frames_after_eos):
#   POCKET_EOS_FRAMES        für alle Texte (leer = Pocket schätzt selbst)
#   POCKET_EOS_FRAMES_SHORT  für kurze Texte unter POCKET_SHORT_WORDS Wörtern (Standard 10)
POCKET_EOS_FRAMES = int(os.environ["POCKET_EOS_FRAMES"]) if os.environ.get("POCKET_EOS_FRAMES", "").strip() else None
POCKET_EOS_FRAMES_SHORT = int(os.environ.get("POCKET_EOS_FRAMES_SHORT", "10") or 10)
POCKET_SHORT_WORDS = int(os.environ.get("POCKET_SHORT_WORDS", "8") or 8)
# fließt in den Zwischenspeicher-Schlüssel: geänderte Einstellung -> neu rechnen
POCKET_PARAMS = (POCKET_STEPS, POCKET_TEMP, POCKET_EOS, POCKET_CONTINUE,
                 POCKET_EOS_FRAMES, POCKET_EOS_FRAMES_SHORT, POCKET_SHORT_WORDS)

CLONE_HINT = (
    "Eigene Stimmen mit Pocket TTS brauchen das Modell mit Stimmklonen: auf "
    "huggingface.co/kyutai/pocket-tts die Bedingungen akzeptieren und HF_TOKEN "
    "in der .env eintragen, dann neu starten."
)


def installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("pocket_tts") is not None


def _import_pocket():
    """pocket_tts setzt beim Import torch.set_num_threads(1) für den ganzen
    Prozess – das würde Kokoro und Qwen ausbremsen. Wert danach zurücksetzen."""
    import torch

    before = torch.get_num_threads()
    from pocket_tts import TTSModel  # noqa: F401

    torch.set_num_threads(before)
    return TTSModel


def fetch(language: str, voice: str | None = None) -> None:
    """Gewichte (und die fertige Stimme bzw. die Stimm-Aufnahme) in einem
    eigenen Prozess mit Internetzugang in den HF-Cache laden."""
    env = dict(os.environ, HF_HUB_OFFLINE="0")
    if not env.get("HF_TOKEN"):
        env.pop("HF_TOKEN", None)
    log.info("Pocket TTS: lade Modell %s herunter (einmalig) …", language)
    t0 = time.perf_counter()
    res = subprocess.run([sys.executable, "-m", "app.pocket", language, *([voice] if voice else [])],
                         env=env,
                         capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        raise RuntimeError(f"Pocket-TTS-Download fehlgeschlagen: {res.stderr.strip()[-500:]}")
    log.info("Pocket TTS: Download fertig in %.0f s (%s)", time.perf_counter() - t0,
             res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "")


_supported: dict[str, bool] = {}


def _supports(model, param: str) -> bool:
    """Kennt die installierte pocket-tts-Version diesen Parameter von
    generate_audio? (teacher_force_previous_chunk z. B. erst nach 4.0.1.)
    Fehlt er: einmal warnen und ohne rechnen."""
    if param not in _supported:
        import inspect

        _supported[param] = param in inspect.signature(model.generate_audio).parameters
        if not _supported[param]:
            log.warning("Pocket: %s wird ignoriert – die installierte pocket-tts-Version "
                        "kennt den Parameter nicht.", param)
    return _supported[param]


def _frames_after_eos(text: str) -> int | None:
    """Wie lange nach dem erkannten Satzende weiter erzeugt wird (je 80 ms).
    Pocket schätzt selbst nur 3–5 Frames – kurze Nachrichten klangen damit
    abgeschnitten (bekannt, kyutai-labs/pocket-tts#174: 14 Frames halfen bei
    kurzen Texten, erzeugten bei langen aber Nachklang). Daher nur für kurze."""
    if POCKET_EOS_FRAMES is not None:
        return POCKET_EOS_FRAMES
    if len(text.split()) < POCKET_SHORT_WORDS:
        return POCKET_EOS_FRAMES_SHORT
    return None


def _prompt_file(ref_wav: str) -> str:
    """Eigene Vorlage für Pocket aufbereiten wie im Kyutai-Training: höchstens
    30 s (wie Pocket selbst), endet auf genau 80 ms Stille (Stille am Ende weg, 20 ms ausblenden,
    80 ms Null anhängen). Endet die Vorlage mitten im Sprechen oder auf langer
    Stille, setzt das Modell sonst die Vorlage fort -> Wortfetzen/Kauderwelsch
    (pocket_tts/data/audio_utils.py: end_on_pause, erst nach 4.0.1 eingebaut).
    Schlägt etwas fehl, bleibt es bei der Originaldatei."""
    try:
        import numpy as np
        import soundfile as sf

        src = Path(ref_wav)
        out = Path(os.environ.get("POCKET_PROMPT_DIR", "/tmp/pocket-prompts"))
        dst = out / f"{src.stem}-{int(src.stat().st_mtime)}.wav"
        if dst.is_file():
            return str(dst)
        audio, sr = sf.read(src, dtype="float32", always_2d=True)
        wav = audio.mean(axis=1)[: int(30 * sr)]  # wie Pocket selbst (truncate)
        frame = max(1, int(0.02 * sr))
        n = len(wav) // frame
        if n == 0:
            return ref_wav
        rms = np.sqrt(np.mean(wav[: n * frame].reshape(n, frame) ** 2, axis=1))
        db = 20 * np.log10(rms + 1e-12)
        loud = np.nonzero(db > db.max() - 35.0)[0]
        end = (int(loud[-1]) + 1) * frame
        wav = wav[:end].copy()
        fade = min(int(0.02 * sr), end)
        wav[end - fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        wav = np.concatenate([wav, np.zeros(int(0.08 * sr), dtype=np.float32)])
        out.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp.wav")
        sf.write(tmp, wav, sr)
        os.replace(tmp, dst)
        log.info("Pocket-Vorlage %s: %.1f s, endet auf Pause", src.stem, len(wav) / sr)
        return str(dst)
    except Exception as exc:
        log.warning("Pocket-Vorlage %s nicht aufbereitet (%s) – nehme Original", ref_wav, exc)
        return ref_wav


class PocketEngine:
    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._states: dict[tuple, object] = {}
        self._fetched: set[str] = set()
        self._lock = threading.Lock()

    def has_cloning(self, language: str) -> bool | None:
        model = self._models.get(language)
        return None if model is None else bool(model.has_voice_cloning)

    def load(self, language: str) -> object:
        with self._lock:
            return self._load_locked(language)

    def _load_locked(self, language: str):
        model = self._models.get(language)
        if model is not None:
            return model
        # Nur ein Pocket-Modell je Prozess: beim Wechsel (z. B. german ->
        # german_24l) das alte samt Stimmen freigeben statt beide zu halten.
        if self._models:
            import gc

            from .tts import release_memory

            self._models.clear()
            self._states.clear()
            gc.collect()
            release_memory()
        TTSModel = _import_pocket()
        # Nur ausdrücklich gesetzte Werte übergeben – sonst exakt der Aufruf wie
        # vorher. Die veröffentlichte Version verträgt z. B. None nicht als
        # "Standard" (ergab Kauderwelsch).
        params: dict = {"language": language}
        if POCKET_TEMP is not None:
            params["temp"] = POCKET_TEMP
        if POCKET_EOS is not None:
            params["eos_threshold"] = POCKET_EOS
        if POCKET_STEPS != 1:
            params["sampler_decode_steps"] = POCKET_STEPS
        try:
            model = TTSModel.load_model(**params)
        except Exception as exc:  # nicht im Cache -> herunterladen
            log.info("Pocket TTS %s nicht im Cache (%s)", language, type(exc).__name__)
            fetch(language)
            self._fetched.add(language)
            model = TTSModel.load_model(**params)
        # Token nachträglich eingetragen? Dann einmal das Modell mit Stimmklonen holen.
        if not model.has_voice_cloning and os.environ.get("HF_TOKEN") and language not in self._fetched:
            try:
                fetch(language)
                model = TTSModel.load_model(**params)
            except Exception as exc:
                log.warning("Pocket TTS: Modell mit Stimmklonen nicht ladbar: %s", exc)
            self._fetched.add(language)
        log.info("Pocket TTS %s geladen (Stimmklonen: %s; Schritte %d, Temperatur %s, "
                 "Satzende %s, Fortsetzen %s)", language,
                 "ja" if model.has_voice_cloning else "nein", POCKET_STEPS,
                 POCKET_TEMP if POCKET_TEMP is not None else "Standard",
                 POCKET_EOS if POCKET_EOS is not None else "Standard",
                 "ja" if POCKET_CONTINUE else "nein")
        self._models[language] = model
        return model

    @staticmethod
    def _voice_source(voice: str) -> tuple[str, bool]:
        """"audio:NAME" = Kyutai-Katalogstimme aus ihrer Original-Aufnahme
        klonen (für Stimmen ohne fertige deutsche Einbettung). Liefert
        (Quelle, braucht Stimmklonen)."""
        if voice.startswith("audio:"):
            from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

            return _ORIGINS_OF_PREDEFINED_VOICES[voice.removeprefix("audio:")], True
        return voice, False

    def _state(self, model, language: str, voice: str, ref_wav: str | None):
        if ref_wav:
            key = (language, ref_wav, Path(ref_wav).stat().st_mtime)
        else:
            key = (language, voice)
        if key not in self._states:
            source, needs_cloning = ((_prompt_file(ref_wav), True) if ref_wav
                                     else self._voice_source(voice))
            if needs_cloning and not model.has_voice_cloning:
                raise RuntimeError(CLONE_HINT)
            t0 = time.perf_counter()
            try:
                state = model.get_state_for_audio_prompt(source, truncate=True)
            except Exception as exc:
                if ref_wav:
                    raise
                # Stimme noch nicht im Cache (Container läuft offline) -> holen
                log.info("Pocket TTS: Stimme %s nicht im Cache (%s)", voice, type(exc).__name__)
                fetch(language, source)
                state = model.get_state_for_audio_prompt(source, truncate=True)
            self._states[key] = state
            log.info("Pocket TTS: Stimme %s vorbereitet in %.1f s",
                     Path(ref_wav).stem if ref_wav else voice, time.perf_counter() - t0)
        return self._states[key]

    def forget(self, ref_wav: str) -> None:
        for key in [k for k in self._states if len(k) == 3 and k[1] == ref_wav]:
            self._states.pop(key, None)

    def synthesize(self, text: str, language: str, voice: str, ref_wav: str | None = None):
        import numpy as np
        import torch

        from .text_normalize import normalize

        # Pocket liest Ziffern, Daten, Uhrzeiten und Abkürzungen nicht selbst
        # (Kauderwelsch) -> vorher ausschreiben. Nur für Pocket; Qwen bleibt unverändert.
        text = normalize(text)

        with self._lock:
            model = self._load_locked(language)
            state = self._state(model, language, voice, ref_wav)
            before = torch.get_num_threads()
            torch.set_num_threads(POCKET_THREADS)
            try:
                t0 = time.perf_counter()
                kwargs: dict = {}
                if POCKET_CONTINUE and _supports(model, "teacher_force_previous_chunk"):
                    kwargs["teacher_force_previous_chunk"] = True
                frames = _frames_after_eos(text)
                if frames is not None and _supports(model, "frames_after_eos"):
                    kwargs["frames_after_eos"] = frames
                audio = model.generate_audio(state, text, **kwargs)
            finally:
                torch.set_num_threads(before)
            samples = audio.detach().cpu().float().numpy().astype(np.float32)
            sr = int(model.sample_rate)
            # kurze Stille anhängen (wie bei Qwen): kein hartes Ende direkt nach dem letzten Laut
            samples = np.concatenate([samples, np.zeros(int(0.15 * sr), dtype=np.float32)])
        took = time.perf_counter() - t0
        dur = len(samples) / sr if sr else 0.0
        log.info("Pocket %s/%s: %.1f s Rechenzeit für %.1f s Audio (RTF %.2f, %d Threads, %d Schritte)",
                 language, Path(ref_wav).stem if ref_wav else voice, took, dur,
                 took / dur if dur else 0.0, POCKET_THREADS, POCKET_STEPS)
        return samples, sr


if __name__ == "__main__":  # Download-Prozess (siehe fetch)
    from pocket_tts import TTSModel
    from pocket_tts.default_parameters import DEFAULT_VOICE_FOR_LANGUAGE

    language = sys.argv[1]
    m = TTSModel.load_model(language=language)
    if len(sys.argv) > 2:
        voice = sys.argv[2]  # bestimmte Stimme/Aufnahme laden
    else:
        voice = DEFAULT_VOICE_FOR_LANGUAGE.get(language.split("_")[0], "juergen")
    m.get_state_for_audio_prompt(voice, truncate=True)
    print(f"ok {language} stimmklonen={'ja' if m.has_voice_cloning else 'nein'} stimme={voice}")
