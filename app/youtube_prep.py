"""YouTube-Video vorbereiten (eigener Prozess, Ausgabe wie beim Stimmenklonen):

  1. Infos holen (yt-dlp -J): Titel, Originalsprache, Untertitel-Spuren
  2. Video in höchstens 720p als MP4 laden (spielt im Browser direkt)
  3. Text besorgen – in dieser Reihenfolge:
       a) echte deutsche Untertitel vom Kanal
       b) Untertitel in der Originalsprache (vom Kanal oder YouTubes
          Spracherkennung) – YouTubes *Übersetzungen* werden bewusst nicht
          genutzt, die blockt YouTube gern mit 429
       c) gibt es nichts (oder blockt YouTube alles): selbst abschreiben mit
          Whisper aus der Tonspur des geladenen Videos
  4. Nicht-deutschen Text lokal übersetzen: Opus-MT (Helsinki-NLP, frei,
     CPU) – direkt Sprache->Deutsch, sonst über Englisch
  5. meta.json mit Sätzen + Startzeiten schreiben

  python -m app.youtube_prep --url URL --out ORDNER
Ausgabe: "SCHRITT: …" (Fortschritt), am Ende "ERGEBNIS {json}".
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .youtube_subs import _cues_json3, _cues_vtt, _lang_name, _sentences

YT_HEIGHT = int(os.environ.get("YT_MAX_HEIGHT", "720") or 720)
YT_WHISPER_MODEL = os.environ.get("YT_WHISPER_MODEL", "small")
TRANSLATE_BATCH = 16


def step(text: str) -> None:
    print(f"SCHRITT: {text}", flush=True)


def ytdlp(*args: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "yt_dlp", "--no-playlist", *args],
                          capture_output=True, text=True, timeout=timeout)


# ------------------------------------------------------------------ Video
def download_video(url: str, out: Path) -> Path:
    step(f"Lade Video (höchstens {YT_HEIGHT}p) …")
    fmt = (f"bv*[height<={YT_HEIGHT}][vcodec^=avc1]+ba[ext=m4a]/"
           f"b[height<={YT_HEIGHT}][ext=mp4]/bv*[height<={YT_HEIGHT}]+ba/b")
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--newline", "-f", fmt,
           "--merge-output-format", "mp4", "-o", str(out / "video.%(ext)s"), url]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = ""
    shown = -10.0
    for line in proc.stdout:
        line = line.strip()
        if line:
            last = line
        m = re.search(r"\[download\]\s+([\d.]+)%", line)
        if m and float(m.group(1)) - shown >= 10:
            shown = float(m.group(1))
            step(f"Lade Video … {shown:.0f} %")
    if proc.wait() != 0:
        raise SystemExit(f"Video nicht ladbar: {last}")
    files = sorted(out.glob("video.*"))
    if not files:
        raise SystemExit("Video nicht ladbar: keine Datei")
    video = files[0]
    if video.suffix != ".mp4":  # notfalls in MP4 umpacken/umwandeln
        step("Wandle Video für den Browser um …")
        mp4 = out / "video.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-movflags", "+faststart",
                        str(mp4)], check=True)
        video.unlink(missing_ok=True)
        video = mp4
    return video


# -------------------------------------------------------------- Untertitel
def _sub_codes(tracks: dict, lang: str) -> list[str]:
    """Spur-Codes zu einer Sprache, beste zuerst ("en", "en-US", "en-orig")."""
    lang = lang.lower()
    codes = [c for c in tracks if c.lower() == lang or c.lower().startswith(lang + "-")]
    # "-orig" = Original-Spracherkennung (keine Übersetzung) zuerst, dann exakt, dann Varianten
    return sorted(codes, key=lambda c: (0 if c.endswith("-orig") else 1 if c.lower() == lang else 2, c))


def fetch_subs(url: str, out: Path, code: str, auto: bool):
    """Eine Untertitel-Spur über yt-dlp laden. None bei Fehler (z. B. 429)."""
    sub_dir = out / "subs"
    sub_dir.mkdir(exist_ok=True)
    for f in sub_dir.glob("*"):
        f.unlink()
    res = ytdlp("--skip-download", "--write-auto-subs" if auto else "--write-subs",
                "--sub-langs", code, "--sub-format", "json3/vtt/best",
                "--retries", "3", "--sleep-subtitles", "2",
                "-o", str(sub_dir / "s.%(ext)s"), url, timeout=300)
    files = sorted(glob.glob(str(sub_dir / "*.json3"))) + sorted(glob.glob(str(sub_dir / "*.vtt")))
    if not files:
        err = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "keine Datei"
        step(f"Untertitel {code} nicht ladbar ({err[:120]})")
        return None
    raw = Path(files[0]).read_text(encoding="utf-8", errors="replace")
    cues = _cues_json3(raw) if files[0].endswith(".json3") else _cues_vtt(raw)
    return cues or None


def whisper_cues(video: Path, out: Path, lang: str | None):
    step(f"Keine nutzbaren Untertitel – schreibe selbst ab (Whisper {YT_WHISPER_MODEL}) …")
    from faster_whisper import WhisperModel

    wav = out / "ton.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vn", "-ac", "1",
                    "-ar", "16000", str(wav)], check=True)
    model = WhisperModel(YT_WHISPER_MODEL, device="cpu", compute_type="int8",
                         cpu_threads=int(os.environ.get("YT_WHISPER_THREADS", "16")))
    segments, info = model.transcribe(str(wav), language=(lang or None), beam_size=5,
                                      vad_filter=True)
    cues = []
    for seg in segments:
        cues.append((seg.start, seg.end, seg.text.strip()))
        if len(cues) % 50 == 0:
            step(f"Abschreiben … {seg.end / 60:.0f} min")
    wav.unlink(missing_ok=True)
    return cues, info.language


# ------------------------------------------------------------- Übersetzen
def _load_mt(repo: str):
    from huggingface_hub import snapshot_download
    from transformers import MarianMTModel, MarianTokenizer

    path = snapshot_download(repo)
    return MarianTokenizer.from_pretrained(path), MarianMTModel.from_pretrained(path).eval()


def translate(texts: list[str], src: str, quiet: bool = False) -> list[str]:
    import torch

    torch.set_num_threads(int(os.environ.get("YT_TRANSLATE_THREADS", "16")))
    src = src.split("-")[0].lower()
    chain: list[tuple] = []
    try:
        if not quiet:
            step(f"Lade Übersetzer {src}→de (beim ersten Mal Download) …")
        chain = [_load_mt(f"Helsinki-NLP/opus-mt-{src}-de")]
    except Exception:
        if src == "en":
            raise
        step(f"Kein direkter Übersetzer {src}→de – übersetze über Englisch …")
        chain = [_load_mt(f"Helsinki-NLP/opus-mt-{src}-en"), _load_mt("Helsinki-NLP/opus-mt-en-de")]
    out = list(texts)
    for tok, model in chain:
        res = []
        for i in range(0, len(out), TRANSLATE_BATCH):
            batch = out[i:i + TRANSLATE_BATCH]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=400)
            with torch.inference_mode():
                gen = model.generate(**enc, num_beams=4, max_new_tokens=400)
            res += tok.batch_decode(gen, skip_special_tokens=True)
            if not quiet:
                step(f"Übersetze … {min(i + TRANSLATE_BATCH, len(out))} von {len(out)} Sätzen")
        out = res
    return out


# ------------------------------------------- Übersetzen mit lokalem LLM
# Deutlich besser als Opus-MT, weil das Modell den Zusammenhang sieht und
# Umgangssprache/zerhackte Untertitel versteht. Nutzt einen OpenAI-
# kompatiblen Server im LAN (llama.cpp, Ollama, …):
#   YT_LLM_URL    z. B. http://192.168.0.190:8080/v1   (leer = aus, Opus-MT)
#   YT_LLM_MODEL  Modellname, wie ihn der Server unter /v1/models nennt
#   YT_LLM_KEY    optional, falls der Server einen API-Schlüssel will
YT_LLM_URL = (os.environ.get("YT_LLM_URL") or os.environ.get("LLM_URL") or "").rstrip("/")
YT_LLM_MODEL = os.environ.get("YT_LLM_MODEL") or os.environ.get("LLM_MODEL") or ""
YT_LLM_KEY = os.environ.get("YT_LLM_KEY") or os.environ.get("LLM_KEY") or ""
LLM_BATCH = 20
LLM_CONTEXT = 4

_PROMPT = (
    "Du übersetzt Untertitel eines Videos aus dem {src} ins Deutsche. Die Übersetzung wird "
    "danach von einer Sprachausgabe vorgelesen.\n"
    "Regeln:\n"
    "- Natürliches, gesprochenes Deutsch, wie ein guter Synchronsprecher es sagen würde – "
    "sinngemäß statt Wort für Wort, Redewendungen passend übertragen.\n"
    "- Untertitel sind oft mitten im Satz zerteilt: Verstehe sie im Zusammenhang, aber gib "
    "JEDE Nummer genau einmal zurück, in derselben Reihenfolge, nichts zusammenlegen, "
    "nichts weglassen.\n"
    "- Möglichst kurz halten (die Zeit im Video ist begrenzt).\n"
    "- Zahlen, Einheiten und Abkürzungen so ausschreiben, wie man sie spricht.\n"
    "- Eigennamen, Produktnamen und Marken nicht übersetzen.\n"
    "- Antworte NUR mit den Zeilen im Format \"Nummer: Übersetzung\", ohne Erklärungen."
)


def _llm(messages: list[dict]) -> str:
    import urllib.request

    body = {"model": YT_LLM_MODEL or "default", "messages": messages, "temperature": 0.3,
            "stream": False,
            # Qwen3 & Co.: kein "Nachdenken" – spart viel Zeit, Übersetzen braucht das nicht
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{YT_LLM_URL}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {YT_LLM_KEY}"} if YT_LLM_KEY else {})})
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.loads(r.read())
    text = data["choices"][0]["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def translate_llm(texts: list[str], src: str) -> list[str] | None:
    """Übersetzen in Blöcken mit etwas Kontext davor. None, wenn der Server
    nicht erreichbar ist (dann Opus-MT). Einzelne kaputte Blöcke -> Opus-MT."""
    if not YT_LLM_URL:
        return None
    src_name = {"en": "Englischen", "fr": "Französischen", "es": "Spanischen", "it": "Italienischen",
                "nl": "Niederländischen", "pl": "Polnischen", "ru": "Russischen",
                "pt": "Portugiesischen", "ja": "Japanischen"}.get(src.split("-")[0], "Fremdsprachigen")
    system = _PROMPT.format(src=src_name)
    out: list[str] = []
    for i in range(0, len(texts), LLM_BATCH):
        batch = texts[i:i + LLM_BATCH]
        ctx = "\n".join(f"(vorher: {t})" for t in out[-LLM_CONTEXT:])
        user = ((f"Zusammenhang (bereits übersetzt, nicht wiederholen):\n{ctx}\n\n" if ctx else "")
                + "Übersetze:\n" + "\n".join(f"{n + 1}: {t}" for n, t in enumerate(batch)))
        try:
            answer = _llm([{"role": "system", "content": system}, {"role": "user", "content": user}])
        except Exception as exc:
            if i == 0:
                step(f"Übersetzungs-LLM nicht erreichbar ({str(exc)[:80]}) – nehme Opus-MT")
                return None
            answer = ""
        got: dict[int, str] = {}
        for line in answer.splitlines():
            m = re.match(r"\s*(\d+)\s*[:.)]\s*(.+)", line)
            if m:
                got[int(m.group(1))] = m.group(2).strip()
        if all(n + 1 in got for n in range(len(batch))):
            out += [got[n + 1] for n in range(len(batch))]
        else:  # Antwort passt nicht 1:1 -> diesen Block klassisch übersetzen
            step(f"LLM-Antwort für Sätze {i + 1}–{i + len(batch)} unvollständig – Opus-MT für diesen Block")
            out += translate(batch, src, quiet=True)
        step(f"Übersetze (LLM) … {min(i + LLM_BATCH, len(texts))} von {len(texts)} Sätzen")
    return out


# ------------------------------------------------------------------- Ablauf
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    step("Hole Video-Infos …")
    res = ytdlp("-J", "--skip-download", args.url, timeout=180)
    if res.returncode != 0:
        msg = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "unbekannter Fehler"
        raise SystemExit(f"Video nicht ladbar: {msg}")
    info = json.loads(res.stdout)
    original = (info.get("language") or "").split("-")[0].lower() or None
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    step(f"„{info.get('title', '')}“ – Original: {_lang_name(original)}")

    video = download_video(args.url, out)

    cues, text_lang, source = None, None, ""
    # a) echte deutsche Untertitel
    for code in _sub_codes(manual, "de"):
        cues = fetch_subs(args.url, out, code, auto=False)
        if cues:
            text_lang, source = "de", "deutsche Untertitel vom Kanal"
            break
    # b) Originalsprache: vom Kanal, sonst YouTubes Spracherkennung
    if not cues and original:
        for tracks, is_auto, label in ((manual, False, "Untertitel vom Kanal"),
                                       (auto, True, "YouTube-Spracherkennung")):
            for code in _sub_codes(tracks, original)[:2]:
                cues = fetch_subs(args.url, out, code, auto=is_auto)
                if cues:
                    text_lang, source = original, f"{label} ({_lang_name(original)})"
                    break
            if cues:
                break
    # c) selbst abschreiben
    if not cues:
        cues, text_lang = whisper_cues(video, out, original)
        source = f"selbst abgeschrieben (Whisper, {_lang_name(text_lang)})"
    if not cues:
        raise SystemExit("Kein Text gefunden – das Video scheint keine Sprache zu enthalten.")

    sentences = _sentences(cues)
    if (text_lang or "de") != "de":
        originals = [s["text"] for s in sentences]
        texts = translate_llm(originals, text_lang)
        how = f"LLM {YT_LLM_MODEL}".strip()
        if texts is None:
            texts, how = translate(originals, text_lang), "Opus-MT"
        for s, t in zip(sentences, texts):
            s["text"] = t
        source += f" → lokal übersetzt ({how})"

    meta = {
        "id": info.get("id"),
        "title": info.get("title") or "",
        "channel": info.get("channel") or info.get("uploader") or "",
        "duration": info.get("duration"),
        "language": _lang_name(original),
        "source": source,
        "video": video.name,
        "sentences": sentences,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    step(f"Fertig: {len(sentences)} Sätze")
    print("ERGEBNIS " + json.dumps({"id": meta["id"], "sentences": len(sentences)}), flush=True)


if __name__ == "__main__":
    main()
