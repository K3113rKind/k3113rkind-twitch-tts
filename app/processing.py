"""Message pre-processing before TTS: emote + URL removal, normalization."""

from __future__ import annotations

import re

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

# Unicode-Emojis (😀 🎉 ❤ …) – mehrere Blöcke, inkl. Variantenselektoren
# und Verbindern, damit zusammengesetzte Emojis komplett verschwinden.
EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # Emoticons, Symbole, Piktogramme, Ergänzungen
    "\U00002600-\U000027BF"  # Verschiedene Symbole, Dingbats
    "\U00002190-\U000021FF"  # Pfeile
    "\U00002B00-\U00002BFF"  # Weitere Symbole
    "\U0000FE00-\U0000FE0F"  # Variantenselektoren
    "\U0001F1E6-\U0001F1FF"  # Flaggen
    "\U0000200D"              # Zero-Width-Joiner
    "\U000020E3"              # Keycap-Zeichen (1️⃣ 2️⃣ …)
    "\U00003030\U0000303D"    # Japanische Sonderzeichen
    "\U000000A9\U000000AE"    # © ®
    "\U00002122"              # ™
    "]+"
)

# Text-Smileys. Bewusst als feste Liste statt breitem Muster: ein
# allgemeines Muster würde sonst auch harmlosen Text zerlegen
# (z.B. "Aufgabe 8)" oder Uhrzeiten). Wiederholungen am Ende sind erlaubt,
# damit auch ":))" oder "xDDD" erfasst werden.
_SMILEYS = [
    ":)", ":-)", "=)", ":(", ":-(", "=(", ";)", ";-)", ":d", ":-d", "=d",
    ":p", ":-p", ";p", ":o", ":-o", ":|", ":-|", ":/", ":-/", ":\\",
    ":'(", ":')", ":*", ":-*", ":3", ":>", ":<", ";d", "xd", "xp",
    "^^", "^_^", "<3", "</3", "-_-", "._.", "o/", "\\o/", "uwu", "owo",
]
SMILEY_RE = re.compile(
    r"(?<!\S)(?:" + "|".join(re.escape(s) for s in _SMILEYS) + r")\S{0,3}(?!\S)",
    re.IGNORECASE,
)
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


def is_speakable(text: str) -> bool:
    """Enthält der Text überhaupt etwas Sprechbares?

    Sicherheitsnetz gegen Nachrichten, die nach dem Bereinigen nur noch aus
    Satz- oder Sonderzeichen bestehen (etwa ein nicht erkanntes Emoji).
    Ohne diese Prüfung würde die Ausgabe nur aus dem Namen bestehen.
    `isalnum` erfasst auch nicht-lateinische Schriften.
    """
    return any(ch.isalnum() for ch in text)


def has_mention(text: str) -> bool:
    """Enthält die Nachricht eine Erwähnung (@name)?

    Wird zum Filtern ganzer Nachrichten genutzt: Ist die Option
    abgeschaltet, werden solche Nachrichten gar nicht erst vorgelesen.
    """
    return bool(MENTION_RE.search(text))


def clean_message(
    text: str,
    emotes_tag: str | None = None,
    read_emotes: bool = False,
    read_smileys: bool = True,
) -> str:
    """Return the speakable remainder of a chat message (may be empty).

    `read_emotes=False` (Default) entfernt Twitch-Emote-Codes.
    `read_smileys=True` (Default) lässt Text-Smileys und Emojis stehen;
    False entfernt sie. Bleibt danach nichts übrig, wird die Nachricht
    nicht vorgelesen (der Aufrufer verwirft leere Texte).
    """
    # /me messages arrive as CTCP ACTION.
    if text.startswith("\x01ACTION") and text.endswith("\x01"):
        text = text[len("\x01ACTION"):-1]
    if not read_emotes:
        text = strip_emotes(text, emotes_tag)
    # URLs zuerst: sonst würde ":/" in "https://" als Smiley gelten.
    text = URL_RE.sub(" ", text)
    if not read_smileys:
        text = SMILEY_RE.sub(" ", text)
        text = EMOJI_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()
