"""Message queue + sequential playback orchestration.

Flow: Twitch message -> filters (bot list, per-user cooldown, cleaning)
-> bounded queue (drop oldest on overflow) -> worker synthesizes via the
TTS engine -> WAV is streamed over WebSocket to the browser -> worker waits
for the browser's "played" ack (or skip / watchdog timeout) before taking
the next message. Guarantees strictly sequential playback, no overlap.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections import deque
from dataclasses import dataclass

from .config import ConfigStore
from .processing import clean_message
from .tts import TTS
from .voices import available_voices, is_available

log = logging.getLogger(__name__)

# Extra grace on top of the audio duration before we assume the browser
# died and move on (keeps the queue alive if no UI tab is open to ack).
ACK_GRACE_SECONDS = 10


@dataclass
class QueuedMessage:
    username: str
    text: str


class Broadcaster:
    """Tracks connected WebSocket UIs and broadcasts JSON messages."""

    def __init__(self) -> None:
        self._clients: set = set()

    def add(self, ws) -> None:
        self._clients.add(ws)

    def remove(self, ws) -> None:
        self._clients.discard(ws)

    @property
    def count(self) -> int:
        return len(self._clients)

    async def send(self, payload: dict) -> None:
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.remove(ws)


class ChatSpeaker:
    def __init__(self, config: ConfigStore, tts: TTS, broadcaster: Broadcaster) -> None:
        self.config = config
        self.tts = tts
        self.broadcaster = broadcaster
        self._queue: deque[QueuedMessage] = deque()
        self._queue_event = asyncio.Event()
        self._last_spoken: dict[str, float] = {}
        self._worker: asyncio.Task | None = None
        self._ack = asyncio.Event()
        self._skip = asyncio.Event()
        self._current_id = 0
        self.dropped_total = 0

    # ---------------------------------------------------------------- intake
    async def on_message(self, username: str, text: str, emotes_tag: str | None) -> None:
        cfg = self.config.as_dict()
        user_key = username.lower()

        if user_key in cfg["bot_blocklist"]:
            return

        now = time.monotonic()
        last = self._last_spoken.get(user_key, 0.0)
        if now - last < cfg["cooldown_seconds"]:
            return

        spoken = clean_message(text, emotes_tag, cfg["read_emotes"])
        if not spoken:
            return

        self._last_spoken[user_key] = now
        self._queue.append(QueuedMessage(username=username, text=spoken))
        # Queue limit: drop oldest instead of building up delay.
        limit = cfg["queue_limit"]
        while len(self._queue) > limit:
            self._queue.popleft()
            self.dropped_total += 1
        self._queue_event.set()
        await self._push_queue_state()

    # ---------------------------------------------------------------- worker
    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run(), name="tts-worker")

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        self._queue.clear()
        self._last_spoken.clear()
        self._skip.set()  # release a potentially waiting playback
        await self.broadcaster.send({"type": "stop_audio"})
        await self._push_queue_state()

    def skip(self) -> None:
        """Abort the currently playing message, jump to the next one."""
        self._skip.set()

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    async def _push_queue_state(self) -> None:
        await self.broadcaster.send({"type": "queue", "size": len(self._queue)})

    async def _run(self) -> None:
        while True:
            while not self._queue:
                self._queue_event.clear()
                await self._queue_event.wait()
            msg = self._queue.popleft()
            await self._push_queue_state()
            try:
                await self._speak(msg)
            except Exception as exc:
                log.error("TTS für Nachricht von %s fehlgeschlagen: %s", msg.username, exc)
                await self.broadcaster.send({"type": "error", "message": str(exc)})

    async def _speak(self, msg: QueuedMessage) -> None:
        cfg = self.config.as_dict()
        voice = cfg["voice"]
        if not is_available(voice):
            usable = available_voices()
            if not usable:
                raise RuntimeError(
                    "Keine Stimmen installiert – bitte install.sh erneut ausführen."
                )
            voice = usable[0]["key"]

        text = f"{msg.username}: {msg.text}" if cfg["read_username"] else msg.text

        result = await asyncio.to_thread(self.tts.synthesize, text, voice, cfg["speed"])

        self._current_id += 1
        self._ack.clear()
        self._skip.clear()
        await self.broadcaster.send(
            {
                "type": "speak",
                "id": self._current_id,
                "username": msg.username,
                "text": msg.text,
                "audio": base64.b64encode(result.wav_bytes).decode("ascii"),
            }
        )

        # Sequential playback: wait until the browser reports completion,
        # skip is pressed, or the watchdog fires (e.g. no UI tab open).
        timeout = result.duration_seconds + ACK_GRACE_SECONDS
        ack_task = asyncio.create_task(self._ack.wait())
        skip_task = asyncio.create_task(self._skip.wait())
        done, pending = await asyncio.wait(
            {ack_task, skip_task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if skip_task in done:
            await self.broadcaster.send({"type": "stop_audio"})

    def notify_played(self, utterance_id: int) -> None:
        if utterance_id == self._current_id:
            self._ack.set()
