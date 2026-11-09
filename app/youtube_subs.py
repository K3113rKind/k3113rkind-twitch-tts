"""Deutsche Untertitel eines YouTube-Videos holen und zu Sätzen zusammenfügen
("YouTube vorlesen" in der Weboberfläche).

yt-dlp liefert die Metadaten (-J) mit allen Untertitel-URLs:
  subtitles           von Menschen erstellte Untertitel
  automatic_captions  automatische Untertitel – darunter auch YouTubes
                      maschinelle Übersetzung in viele Sprachen ("de")
Genommen wird bevorzugt ein echter deutscher Untertitel, sonst der
automatische deutsche (bei englischen Videos = automatisch übersetzt).
Format json3: Ereignisse mit Startzeit (ms) und Textstücken, ohne die
rollenden Wiederholungen der VTT-Variante.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import urllib.request

log = logging.getLogger(__name__)

LANGS = {"de": "Deutsch", "en": "Englisch", "fr": "Französisch", "es": "Spanisch", "it": "Italienisch",
         "nl": "Niederländisch", "pl": "Polnisch", "ru": "Russisch", "pt": "Portugiesisch",
         "ja": "Japanisch", "ko": "Koreanisch", "zh": "Chinesisch", "tr": "Türkisch"}

MAX_SENTENCE_CHARS = 220   # ohne Satzzeichen spätestens hier trennen
GAP_SPLIT_S = 1.5          # längere Sprechpause im Video = neuer Satz
_END = re.compile(r"[.!?…]['\"»«)\]]*$")
_ABBR = re.compile(r"(?:^|\s)(?:z|d|u|o|v|bzw|ca|usw|etc|vgl|Nr|Dr|Prof|St|Mio|Mrd|evtl|ggf|inkl)\.$", re.I)


def _lang_name(code: str | None) -> str:
    if not code:
        return "unbekannt"
    return LANGS.get(code.split("-")[0].lower(), code)


def _pick(tracks: dict) -> tuple[str, dict] | None:
    """Deutsche Spur im Format json3 (sonst vtt) wählen."""
    # "de-orig" (Original-Spracherkennung) nur, wenn es sonst nichts gibt
    for code in sorted(tracks, key=lambda c: (c.endswith("-orig"), c != "de", c != "de-DE", c)):
        if code == "de" or code.lower().startswith("de-"):
            fmts = {f.get("ext"): f for f in tracks[code] if f.get("url")}
            for ext in ("json3", "vtt"):
                if ext in fmts:
                    return code, fmts[ext]
    return None


def _cues_json3(raw: str) -> list[tuple[float, float, str]]:
    data = json.loads(raw)
    cues = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        if not text:
            continue
        start = ev.get("tStartMs", 0) / 1000.0
        dur = ev.get("dDurationMs", 0) / 1000.0
        cues.append((start, start + dur, text))
    return cues


_TS = re.compile(r"(\d+):(\d{2}):(\d{2})[.,](\d{3})|(\d{2}):(\d{2})[.,](\d{3})")


def _ts(s: str) -> float:
    m = _TS.search(s)
    if not m:
        return 0.0
    if m.group(1):
        h, mi, se, ms = (int(m.group(i)) for i in (1, 2, 3, 4))
    else:
        h, (mi, se, ms) = 0, (int(m.group(i)) for i in (5, 6, 7))
    return h * 3600 + mi * 60 + se + ms / 1000.0


def _cues_vtt(raw: str) -> list[tuple[float, float, str]]:
    cues, last = [], ""
    for block in re.split(r"\n\s*\n", raw):
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        a, b = lines[idx].split("-->", 1)
        text = " ".join(re.sub(r"<[^>]+>", "", ln) for ln in lines[idx + 1:]).strip()
        if not text or text == last:  # rollende Wiederholungen
            continue
        last = text
        cues.append((_ts(a), _ts(b), text))
    return cues


def _sentences(cues: list[tuple[float, float, str]]) -> list[dict]:
    """Untertitel-Häppchen zu ganzen Sätzen. Startzeit = Beginn des ersten
    Häppchens. Getrennt wird am Satzende, an längeren Sprechpausen und
    spätestens nach MAX_SENTENCE_CHARS (für Untertitel ohne Satzzeichen)."""
    out: list[dict] = []
    cur: list[str] = []
    start = None
    prev_end = None
    speaker = 0  # Sprecherwechsel: YouTube markiert ihn mit ">>", Dialoge mit "- "

    def flush():
        nonlocal cur, start
        text = re.sub(r"\s+", " ", " ".join(cur)).strip()
        if text and re.search(r"\w", text):
            out.append({"t": round(start or 0.0, 2), "text": text, "sp": speaker})
        cur, start = [], None

    for a, b, text in cues:
        text = re.sub(r"\[[^\]]*\]", " ", text)  # [Musik], [Applaus] gar nicht erst aufnehmen
        if not text.strip():
            continue
        if cur and prev_end is not None and a - prev_end > GAP_SPLIT_S:
            flush()
        if re.match(r"\s*-\s+\S", text):  # Dialog-Strich am Zeilenanfang
            if cur:
                flush()
            speaker += 1
            text = re.sub(r"^\s*-\s+", "", text)
        for word in text.split():
            if word.startswith(">>"):  # Sprecherwechsel mitten im Häppchen
                if cur:
                    flush()
                speaker += 1
                word = word[2:]
                if not word:
                    continue
            if start is None:
                start = a
            cur.append(word)
            joined_len = sum(len(w) + 1 for w in cur)
            is_end = (_END.search(word) and not _ABBR.search(" " + word)
                      and not re.fullmatch(r"\w\.", word)          # z. B., A. Müller
                      and not re.fullmatch(r"\d+\.", word))        # 3. Platz
            if is_end or joined_len > MAX_SENTENCE_CHARS:
                flush()
        prev_end = b
    if cur:
        flush()
    return out


def _download(url: str, vid: str, code: str, fmt: dict, auto: bool) -> tuple[str, str]:
    """Untertitel-Datei holen. Erst direkt; antwortet YouTube mit 429 (Too
    Many Requests – passiert vor allem bei automatisch übersetzten
    Untertiteln), lädt yt-dlp sie selbst: mit Wiederholungen und Pausen und
    dem YouTube-Zugang, den es auch für die Videos nutzt."""
    try:
        req = urllib.request.Request(fmt["url"], headers={"User-Agent": "Mozilla/5.0"})
        return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace"), fmt.get("ext", "vtt")
    except Exception as exc:
        log.info("Untertitel direkt nicht ladbar (%s) – versuche es über yt-dlp", exc)
    import glob
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, "-m", "yt_dlp", "--skip-download", "--no-playlist",
               "--write-auto-subs" if auto else "--write-subs", "--sub-langs", code,
               "--sub-format", "json3/vtt/best", "--retries", "5", "--extractor-retries", "3",
               "--sleep-subtitles", "3", "-o", f"{tmp}/%(id)s.%(ext)s", url]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        files = sorted(glob.glob(f"{tmp}/*.json3")) + sorted(glob.glob(f"{tmp}/*.vtt"))
        if not files:
            err = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "keine Datei"
            if "429" in err:
                raise RuntimeError("YouTube blockt den Abruf der Untertitel gerade (429 Too Many "
                                   "Requests). Das passiert vor allem bei automatisch übersetzten "
                                   "Untertiteln – in ein paar Minuten erneut versuchen.")
            raise RuntimeError(f"Untertitel nicht ladbar: {err}")
        path = files[0]
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read(), "json3" if path.endswith(".json3") else "vtt"


def load(url: str) -> dict:
    """Metadaten + deutsche Untertitel holen (blockierend, im Thread aufrufen)."""
    res = subprocess.run([sys.executable, "-m", "yt_dlp", "-J", "--no-playlist", "--skip-download", url],
                         capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        msg = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "unbekannter Fehler"
        raise RuntimeError(f"Video nicht ladbar: {msg}")
    info = json.loads(res.stdout)
    original = info.get("language")
    picked = _pick(info.get("subtitles") or {})
    source = "Untertitel (vom Kanal)"
    if picked is None:
        picked = _pick(info.get("automatic_captions") or {})
        source = ("automatische Untertitel" if (original or "").startswith("de")
                  else f"automatisch übersetzt aus {_lang_name(original)}")
    if picked is None:
        raise RuntimeError("Für dieses Video gibt es keine deutschen Untertitel "
                           "(auch keine automatische Übersetzung).")
    code, fmt = picked
    auto = source != "Untertitel (vom Kanal)"
    raw, ext = _download(url, info.get("id") or "video", code, fmt, auto)
    cues = _cues_json3(raw) if ext == "json3" else _cues_vtt(raw)
    sentences = _sentences(cues)
    if not sentences:
        raise RuntimeError("Die deutschen Untertitel sind leer.")
    log.info("YouTube %s: %d Untertitel -> %d Sätze (%s, %s)", info.get("id"), len(cues),
             len(sentences), code, source)
    return {
        "id": info.get("id"),
        "title": info.get("title") or "",
        "channel": info.get("channel") or info.get("uploader") or "",
        "duration": info.get("duration"),
        "language": _lang_name(original),
        "source": source,
        "sentences": sentences,
    }
