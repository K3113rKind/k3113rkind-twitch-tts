"""Durchgehender Audio-Stream (Webradio-Prinzip) für alle Wiedergabegeräte.

Warum: iOS (Safari und damit auch Brave/Firefox/Chrome auf dem iPhone, alle
nutzen WebKit) pausiert reines WebAudio samt JavaScript, sobald die Seite im
Hintergrund liegt oder das Display gesperrt wird. Am Leben bleibt nur ein
tatsächlich laufendes Media-Element. Deshalb liefert der Server einen
endlosen MP3-Stream unter /stream, den der Browser per <audio> abspielt –
genau wie ein Internetradio, das auch bei gesperrtem iPhone weiterläuft.

Aufbau:
- Ein Takt-Task erzeugt alle 100 ms einen PCM-Block (24 kHz, mono).
- Läuft gerade eine Ansage, kommt der Block aus ihr, sonst Stille.
- Darunter liegt immer der einstellbare Hintergrundton (60 Hz) und – bei
  Stille – der periodische Weckimpuls für Lautsprecher (40 Hz alle 45 s).
- Jeder Hörer bekommt die Blöcke in eine eigene Queue und kodiert sie mit
  einem eigenen MP3-Encoder (Neueinsteiger starten sauber mit einem
  vollständigen Frame, statt mitten in einen fremden Bitstrom zu fallen).

Die Warteschlange (player.py) wartet nicht mehr auf eine Rückmeldung des
Browsers, sondern darauf, dass der Server die Ansage komplett in den Stream
geschrieben hat. Der Hörer hört sie um seinen Puffer (ca. 1–2 s) versetzt.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np

from .config import ConfigStore

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
BLOCK_SECONDS = 0.1
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_SECONDS)
MP3_BITRATE = 64  # kbit/s, mono Sprache – mehr bringt hörbar nichts

# Vorlauf beim Verbinden: so viel Stille geht sofort raus, damit der Browser
# direkt losspielt und einen Puffer gegen kurze Aussetzer hat. Mehr Vorlauf
# heißt stabiler, aber auch mehr Verzögerung zwischen Chat und Ton.
PREROLL_SECONDS = 1.5

# Hängt ein Hörer (langsames Netz) so weit hinterher, werden seine ältesten
# Blöcke verworfen statt endlos Verzögerung aufzubauen.
LISTENER_MAX_BLOCKS = int(5 / BLOCK_SECONDS)

# Hinkt der Takt hinterher (Event-Loop kurz blockiert), wird bis zu dieser
# Menge nachgeliefert; darüber hinaus wird neu synchronisiert.
MAX_CATCHUP_SECONDS = 2.0

# Hintergrundton: hält Tab/Media-Session als "spielt Ton" markiert.
KEEPALIVE_FREQ = 60.0

# Weckimpuls für Lautsprecher, die bei Stille in den Standby gehen. Werte wie
# bisher im Browser (orientiert an KeepSpeekerAwake, halbierter Pegel).
# Keine hohen Frequenzen: 18–19 kHz hören Kinder noch.
WAKE_INTERVAL_SECONDS = 45.0
WAKE_LENGTH_SECONDS = 1.5
WAKE_FADE_SECONDS = 0.25
WAKE_LEVEL = 0.005
WAKE_FREQ = 40.0


def _make_encoder():
    import lameenc

    enc = lameenc.Encoder()
    enc.set_bit_rate(MP3_BITRATE)
    enc.set_in_sample_rate(SAMPLE_RATE)
    enc.set_channels(1)
    enc.set_quality(5)  # 2 = beste, 7 = schnellste; 5 reicht für Sprache
    return enc


def _to_pcm16(block: np.ndarray) -> bytes:
    return (np.clip(block, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class _Utterance:
    __slots__ = ("samples", "pos", "done")

    def __init__(self, samples: np.ndarray, done: asyncio.Future) -> None:
        self.samples = samples
        self.pos = 0
        self.done = done


class AudioStream:
    def __init__(self, config: ConfigStore) -> None:
        self.config = config
        self._listeners: set[asyncio.Queue[bytes]] = set()
        self._current: _Utterance | None = None
        self._task: asyncio.Task | None = None
        self._sample_clock = 0          # fortlaufender Zähler für Phasen
        self._idle_samples = 0          # Stille seit der letzten Ansage/Impuls
        self._wake_pos: int | None = None  # Position im laufenden Weckimpuls

    # ------------------------------------------------------------ Hörer
    @property
    def listener_count(self) -> int:
        return len(self._listeners)

    async def listen(self) -> AsyncIterator[bytes]:
        """MP3-Bytes für einen neuen Hörer, endlos."""
        self.start()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=LISTENER_MAX_BLOCKS)
        encoder = _make_encoder()
        self._listeners.add(queue)
        log.info("Stream-Hörer verbunden (%d aktiv)", len(self._listeners))
        try:
            preroll = np.zeros(int(SAMPLE_RATE * PREROLL_SECONDS), dtype=np.float32)
            yield bytes(encoder.encode(_to_pcm16(preroll)))
            while True:
                pcm = await queue.get()
                data = encoder.encode(pcm)
                if data:
                    yield bytes(data)
        finally:
            self._listeners.discard(queue)
            log.info("Stream-Hörer getrennt (%d aktiv)", len(self._listeners))

    # --------------------------------------------------------- Ansagen
    def play(self, samples: np.ndarray) -> asyncio.Future:
        """Ansage in den Stream legen. Das Future wird erfüllt, sobald sie
        vollständig ausgegeben (oder abgebrochen) wurde."""
        self.start()
        self.stop_current()
        done = asyncio.get_running_loop().create_future()
        self._current = _Utterance(np.asarray(samples, dtype=np.float32), done)
        return done

    def stop_current(self) -> None:
        cur = self._current
        self._current = None
        if cur and not cur.done.done():
            cur.done.set_result(False)

    @property
    def speaking(self) -> bool:
        return self._current is not None

    # ------------------------------------------------------------ Takt
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(
                self._run(), name="audio-stream-clock"
            )

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        next_at = loop.time()
        while True:
            now = loop.time()
            if now - next_at > MAX_CATCHUP_SECONDS:
                log.warning("Audio-Takt %.1f s im Rückstand – synchronisiere neu", now - next_at)
                next_at = now
            try:
                pcm = _to_pcm16(self._next_block())
            except Exception:  # der Takt darf nie sterben
                log.exception("Fehler beim Erzeugen des Audio-Blocks")
                pcm = bytes(BLOCK_SAMPLES * 2)
            for q in list(self._listeners):
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(pcm)
            next_at += BLOCK_SECONDS
            delay = next_at - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)  # aufholen, aber andere Tasks nicht aushungern

    def _next_block(self) -> np.ndarray:
        cfg = self.config.as_dict()
        block = np.zeros(BLOCK_SAMPLES, dtype=np.float32)
        t = (self._sample_clock + np.arange(BLOCK_SAMPLES)) / SAMPLE_RATE

        # Ansage
        cur = self._current
        if cur is not None:
            chunk = cur.samples[cur.pos:cur.pos + BLOCK_SAMPLES]
            block[:len(chunk)] += chunk * float(cfg["volume"])
            cur.pos += BLOCK_SAMPLES
            if cur.pos >= len(cur.samples):
                self._current = None
                if not cur.done.done():
                    cur.done.set_result(True)
            self._idle_samples = 0
            self._wake_pos = None

        # Hintergrundton (0 = aus)
        level = float(cfg["keepalive_level"])
        if level > 0:
            block += level * np.sin(2 * np.pi * KEEPALIVE_FREQ * t, dtype=np.float32)

        # Weckimpuls nur in Sprechpausen
        if cur is None:
            self._idle_samples += BLOCK_SAMPLES
            if (
                self._wake_pos is None
                and cfg["keep_speakers_awake"]
                and self._idle_samples >= WAKE_INTERVAL_SECONDS * SAMPLE_RATE
            ):
                self._wake_pos = 0
            if self._wake_pos is not None:
                block += self._wake_block(t)

        self._sample_clock += BLOCK_SAMPLES
        return block

    def _wake_block(self, t: np.ndarray) -> np.ndarray:
        length = int(WAKE_LENGTH_SECONDS * SAMPLE_RATE)
        fade = int(WAKE_FADE_SECONDS * SAMPLE_RATE)
        idx = self._wake_pos + np.arange(BLOCK_SAMPLES)
        # Trapez-Hüllkurve: weich ein- und ausblenden gegen Knacken
        env = np.minimum(1.0, np.minimum(idx / fade, (length - idx) / fade))
        env = np.clip(env, 0.0, 1.0).astype(np.float32)
        self._wake_pos += BLOCK_SAMPLES
        if self._wake_pos >= length:
            self._wake_pos = None
            self._idle_samples = 0
        return WAKE_LEVEL * env * np.sin(2 * np.pi * WAKE_FREQ * t, dtype=np.float32)
