"""Textbox: eigenen Text vorlesen (nur ohne laufenden Chat).

Der Text wird an Satzgrenzen in Abschnitte geteilt (nie mitten im Satz).
"Direkt vorlesen" startet, sobald der erste Abschnitt fertig ist, und rechnet
die nächsten währenddessen. "Verzögert vorlesen" rechnet erst die gewählte
Anzahl Abschnitte (oder alles) und liest dann ohne Pausen vor.
"""

from __future__ import annotations

import asyncio
import logging
import re

from . import loudness
from .voices import DEFAULT_VOICE, available_voices, is_available

log = logging.getLogger(__name__)

MAX_CHARS = 1000
SECTION_CHARS = 220  # Abschnitte bis etwa so lang, aus ganzen Sätzen
PAUSE_S = 0.25       # kurze Pause zwischen zwei Abschnitten

_SENTENCE = re.compile(r"(?<=[.!?…])\s+|\n\s*\n")


def sections(text: str) -> list[str]:
    """Ganze Sätze zu Abschnitten bis ~SECTION_CHARS zusammenfassen. Ein
    einzelner längerer Satz bleibt ein eigener Abschnitt (wird nie geteilt)."""
    sentences = [s.strip() for s in _SENTENCE.split(text) if s and s.strip()]
    out: list[str] = []
    for s in sentences:
        if out and len(out[-1]) + 1 + len(s) <= SECTION_CHARS:
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)
    return out


class TextReader:
    def __init__(self, config, tts, broadcaster, audio) -> None:
        self.config = config
        self.tts = tts
        self.broadcaster = broadcaster
        self.audio = audio
        self._task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def cancel(self) -> None:
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            self.audio.stop_current()
            await self.broadcaster.send({"type": "stop_audio"})

    async def start(self, text: str, prebuffer: int) -> int:
        """prebuffer: 0 = direkt, 1–6 = so viele Abschnitte vorab, -1 = alles.
        Gibt die Zahl der Abschnitte zurück."""
        parts = sections(text[:MAX_CHARS])
        if not parts:
            raise ValueError("Kein Text.")
        await self.cancel()
        self._task = asyncio.create_task(self._run(parts, prebuffer))
        return len(parts)

    async def _run(self, parts: list[str], prebuffer: int) -> None:
        import numpy as np

        cfg = self.config.as_dict()
        voice = cfg["voice"]
        if not is_available(voice):
            usable = available_voices()
            voice = usable[0]["key"] if usable else DEFAULT_VOICE
        speed = cfg["speed"]
        n = len(parts)
        ready: list[asyncio.Future] = [asyncio.get_running_loop().create_future() for _ in parts]

        async def produce() -> None:
            for i, part in enumerate(parts):
                try:
                    r = await asyncio.to_thread(self.tts.synthesize, part, voice, speed)
                    sr = getattr(r, "sample_rate", 24000)
                    samples = loudness.normalize(r.samples, sr)
                    gap = np.zeros(int(sr * PAUSE_S), dtype=np.float32)
                    ready[i].set_result(np.concatenate([samples, gap]) if i < n - 1 else samples)
                except Exception as exc:
                    for f in ready[i:]:
                        if not f.done():
                            f.set_exception(exc)
                    return

        producer = asyncio.create_task(produce())
        try:
            need = n if prebuffer < 0 else min(n, max(1, prebuffer))
            if need > 1:
                await self.broadcaster.send({"type": "speak", "id": 0, "username": "Text",
                                             "text": f"Bereite {need} von {n} Abschnitten vor …"})
            for f in ready[:need]:
                await asyncio.shield(f)
            for i, part in enumerate(parts):
                samples = await asyncio.shield(ready[i])
                await self.broadcaster.send({"type": "speak", "id": 0, "username": "Text",
                                             "text": part})
                finished = await self.audio.play(samples)
                if finished is False:  # abgebrochen (Überspringen, Chat gestartet …)
                    break
            await self.broadcaster.send({"type": "done", "id": 0})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Textbox: Vorlesen fehlgeschlagen: %s", exc)
            await self.broadcaster.send({"type": "error", "message": f"Vorlesen fehlgeschlagen: {exc}"})
        finally:
            producer.cancel()
            for f in ready:  # nicht abgeholte Fehler nicht als "never retrieved" melden
                if f.done() and not f.cancelled():
                    f.exception()
