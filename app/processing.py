"""Message pre-processing before TTS: emote + URL removal, normalization."""

from __future__ import annotations

import re

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# Erwähnung: @name am Wortanfang. Die Bedingung davor verhindert, dass
# E-Mail-Adressen (mail@example.com) fälschlich als Erwähnung zählen.
MENTION_RE = re.compile(r"(?<![\w.@])@\w+")
WS_RE = re.compile(r"\s+")


def strip_emotes(text: str, emotes_tag: str | None) -> str:
    """Remove Twitch emote codes using the IRC `emotes=` tag.

    Tag format: "25:0-4,12-16/1902:6-10" – ranges are inclusive indices into
    the message measured in Unicode code points. Requires the
    `twitch.tv/tags` capability (which we request); without the tag the text
    is returned unchanged.
    """
    if not emotes_tag:
        return text
    ranges: list[tuple[int, int]] = []
    for emote in emotes_tag.split("/"):
        try:
            _, positions = emote.split(":", 1)
        except ValueError:
            continue
        for pos in positions.split(","):
            try:
                start, end = pos.split("-", 1)
                ranges.append((int(start), int(end)))
            except ValueError:
                continue
    if not ranges:
        return text
    chars = list(text)
    for start, end in ranges:
        for i in range(start, min(end + 1, len(chars))):
            chars[i] = " "
    return "".join(chars)


def has_mention(text: str) -> bool:
    """Enthält die Nachricht eine Erwähnung (@name)?

    Wird zum Filtern ganzer Nachrichten genutzt: Ist die Option
    abgeschaltet, werden solche Nachrichten gar nicht erst vorgelesen.
    """
    return bool(MENTION_RE.search(text))


def clean_message(
    text: str, emotes_tag: str | None = None, read_emotes: bool = False
) -> str:
    """Return the speakable remainder of a chat message (may be empty).

    `read_emotes=False` (Default) entfernt Emote-Codes; True lässt sie
    stehen, damit die TTS sie mit vorliest.
    """
    # /me messages arrive as CTCP ACTION.
    if text.startswith("\x01ACTION") and text.endswith("\x01"):
        text = text[len("\x01ACTION"):-1]
    if not read_emotes:
        text = strip_emotes(text, emotes_tag)
    text = URL_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()
