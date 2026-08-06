"""Minimal async Twitch IRC client (read-only).

Connects anonymously (justinfan-nick, no auth token needed) to
irc.chat.twitch.tv over TLS and joins one channel. Requests the
`twitch.tv/tags` capability so PRIVMSGs carry display-name and emote
positions (used for clean TTS output).

Note: Twitch is gradually migrating chat from IRC to EventSub; the IRC
interface still works as of 2026, but this client is isolated behind a
simple callback so it can be replaced by an EventSub transport later
without touching the rest of the application.
"""

from __future__ import annotations

import asyncio
import logging
import random
import ssl
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

HOST = "irc.chat.twitch.tv"
PORT = 6697

# callback(username, message_text, emotes_tag)
MessageHandler = Callable[[str, str, str | None], Awaitable[None]]


def _parse_tags(raw: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for part in raw.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            tags[key] = value
    return tags


class TwitchChatClient:
    def __init__(self, channel: str, on_message: MessageHandler) -> None:
        self.channel = channel.lstrip("#").lower()
        self.on_message = on_message
        self.connected = False
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    # ---------------------------------------------------------------- public
    def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="twitch-irc")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.connected = False

    # --------------------------------------------------------------- internal
    async def _run(self) -> None:
        backoff = 2
        while not self._stop.is_set():
            try:
                await self._session()
                backoff = 2  # clean disconnect -> quick retry
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("IRC-Verbindung verloren (%s), Reconnect in %ss", exc, backoff)
            finally:
                self.connected = False
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    async def _session(self) -> None:
        ctx = ssl.create_default_context()
        reader, writer = await asyncio.open_connection(HOST, PORT, ssl=ctx)
        try:
            nick = f"justinfan{random.randint(10000, 99999)}"

            async def send(line: str) -> None:
                writer.write((line + "\r\n").encode("utf-8"))
                await writer.drain()

            # Anonymous read-only join: no PASS/token required. A stored
            # OAuth token (config) is intentionally NOT used here yet – it is
            # reserved for future write features.
            await send("CAP REQ :twitch.tv/tags")
            await send(f"NICK {nick}")
            await send(f"JOIN #{self.channel}")
            log.info("Verbunden mit #%s als %s (anonym, read-only)", self.channel, nick)
            self.connected = True

            while not self._stop.is_set():
                raw = await asyncio.wait_for(reader.readline(), timeout=360)
                if not raw:
                    raise ConnectionError("Server hat die Verbindung geschlossen")
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    continue
                await self._handle_line(line, send)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_line(self, line: str, send) -> None:
        tags: dict[str, str] = {}
        if line.startswith("@"):
            raw_tags, _, line = line.partition(" ")
            tags = _parse_tags(raw_tags[1:])

        if line.startswith("PING"):
            await send(line.replace("PING", "PONG", 1))
            return

        # :nick!nick@nick.tmi.twitch.tv PRIVMSG #channel :text
        prefix = ""
        if line.startswith(":"):
            prefix, _, line = line.partition(" ")
        command, _, rest = line.partition(" ")

        if command == "PRIVMSG":
            _, _, text = rest.partition(" :")
            username = tags.get("display-name") or prefix[1:].split("!", 1)[0]
            await self.on_message(username, text, tags.get("emotes") or None)
        elif command == "RECONNECT":
            raise ConnectionError("Server verlangt RECONNECT")
