"""Eigene Stimme für Qwen-Stimmklonen erzeugen – komplett automatisch.

Läuft im Wegwerf-Container aus tools/stimme-klonen.sh:
  1. Ton holen (YouTube-Adresse per yt-dlp, oder Datei unter /quelle)
  2. Stimme von Musik trennen (Demucs htdemucs)
  3. Mit Wort-Zeitstempeln abschreiben (faster-whisper)
  4. Besten Abschnitt von 8–12 s automatisch wählen: möglichst an Satz-
     grenzen, ohne lange Pausen, mit sicher erkannten Wörtern
  5. NAME.wav (24 kHz mono) + NAME.txt in den Referenzordner schreiben
  6. Qwen-Base-Modelle in den Hugging-Face-Cache laden, falls sie fehlen
"""

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

SRC = os.environ["QUELLE"].strip()
NAME = os.environ["NAME"].strip()
REF_DIR = Path("/ref")
WORK = Path("/arbeit") / NAME
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
BASE_MODELS = ["Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"]

MIN_S, MAX_S, IDEAL_S = 8.0, 12.0, 10.0
PAD_S = 0.15  # Luft vor/nach dem Abschnitt, damit nichts abgeschnitten klingt


def run(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


def step(text: str) -> None:
    print(f"\n== {text}", flush=True)


# ------------------------------------------------------------------ 1. Ton
def ensure_deno() -> None:
    """yt-dlp braucht für YouTube eine JavaScript-Laufzeit (Deno >= 2.3)."""
    if shutil.which("deno"):
        return
    url = "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip"
    zpath = Path("/tmp/deno.zip")
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extract("deno", "/usr/local/bin")
    os.chmod("/usr/local/bin/deno", 0o755)


def fetch_source() -> Path:
    existing = sorted(WORK.glob("quelle.*"))
    if existing:
        print(f"Quelle schon vorhanden ({existing[0].name}), lade nicht erneut.")
        return existing[0]
    if SRC.startswith(("http://", "https://")):
        step("Ton von YouTube laden")
        ensure_deno()
        run("yt-dlp", "--no-playlist", "-f", "bestaudio", "-x", "--audio-format", "wav",
            "-o", str(WORK / "quelle.%(ext)s"), SRC)
    else:
        src = Path("/quelle") / Path(SRC).name
        if not src.is_file():
            sys.exit(f"Datei nicht gefunden: {SRC}")
        shutil.copy(src, WORK / f"quelle{src.suffix}")
    return sorted(WORK.glob("quelle.*"))[0]


# ------------------------------------------------------------ 2. Trennen
def separate(source: Path) -> Path:
    vocals = WORK / "stimme_komplett.wav"
    if vocals.exists():
        return vocals
    step("Stimme von Musik trennen (Demucs)")
    run("ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-ac", "2", "-ar", "44100", "/tmp/in.wav")
    run(sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems=vocals", "-o", "/tmp/sep", "/tmp/in.wav")
    run("ffmpeg", "-loglevel", "error", "-y", "-i", "/tmp/sep/htdemucs/in/vocals.wav",
        "-ac", "1", "-ar", "24000", str(vocals))
    return vocals


# ---------------------------------------------------------- 3. Abschrift
def transcribe(path: Path):
    step(f"Abschreiben (Whisper {WHISPER_MODEL})")
    from faster_whisper import WhisperModel

    import soundfile as sf

    # Audio selbst als 16-kHz-Array übergeben: faster-whispers eigener Decoder
    # (PyAV) passt je nach Versionskombination nicht zusammen.
    run("ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-ac", "1", "-ar", "16000", "/tmp/whisper.wav")
    audio, _ = sf.read("/tmp/whisper.wav", dtype="float32")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio, language="de", beam_size=5, word_timestamps=True)
    words = [w for s in segments for w in (s.words or [])]
    for s_word in words:
        s_word.word = s_word.word.strip()
    return [w for w in words if w.word]


# ------------------------------------------------- 4. Abschnitt auswählen
def ends_sentence(word: str) -> bool:
    return word.endswith((".", "!", "?", "…"))


def pick_window(words):
    """Bewertet alle Wortfolgen von 8–12 s und nimmt die beste."""
    best, best_score = None, float("-inf")
    for i in range(len(words)):
        for j in range(i, len(words)):
            dur = words[j].end - words[i].start
            if dur > MAX_S:
                break
            if dur < MIN_S:
                continue
            seq = words[i:j + 1]
            gaps = [b.start - a.end for a, b in zip(seq, seq[1:])]
            score = 0.0
            score += 3 if i == 0 or ends_sentence(words[i - 1].word) else 0  # beginnt am Satzanfang
            score += 3 if ends_sentence(words[j].word) else (1 if words[j].word.endswith(",") else 0)
            score -= abs(dur - IDEAL_S) * 0.5                                # nahe 10 s
            score -= sum(max(0.0, g - 0.6) for g in gaps) * 2                # lange Pausen (Musik?)
            score += 4 * (sum(w.probability for w in seq) / len(seq))        # sicher erkannt
            if score > best_score:
                best, best_score = (i, j), score
    return best


def cut(vocals: Path, words, i: int, j: int) -> None:
    prev_end = words[i - 1].end if i > 0 else 0.0
    next_start = words[j + 1].start if j + 1 < len(words) else words[j].end + PAD_S
    start = max(prev_end, words[i].start - PAD_S, 0.0)
    end = min(next_start, words[j].end + PAD_S)
    dur = end - start
    text = " ".join(w.word for w in words[i:j + 1])

    REF_DIR.mkdir(parents=True, exist_ok=True)
    out_wav, out_txt = REF_DIR / f"{NAME}.wav", REF_DIR / f"{NAME}.txt"
    fade = f"afade=t=in:d=0.03,afade=t=out:st={max(dur - 0.05, 0):.3f}:d=0.05,loudnorm=I=-20:TP=-2"
    run("ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(vocals),
        "-af", fade, "-ac", "1", "-ar", "24000", str(out_wav))
    out_txt.write_text(text + "\n", encoding="utf-8")
    shutil.copy(out_wav, WORK / out_wav.name)
    shutil.copy(out_txt, WORK / out_txt.name)

    print(f"\nGewählt: {start:.1f}–{end:.1f} s ({dur:.1f} s) aus der Aufnahme")
    print(f"Text:    {text}")


# ---------------------------------------------------- 6. Base-Modelle
def ensure_models() -> None:
    from huggingface_hub import snapshot_download

    for repo in BASE_MODELS:
        step(f"Modell {repo} (wird nur beim ersten Mal geladen)")
        snapshot_download(repo)


# ---------------------------------------------------------------- Ablauf
WORK.mkdir(parents=True, exist_ok=True)
vocals = separate(fetch_source())
words = transcribe(vocals)
if not words:
    sys.exit("Whisper hat keine Sprache gefunden – ist in der Aufnahme überhaupt jemand zu hören?")

total = words[-1].end - words[0].start
print("\nGesamte Abschrift:\n  " + " ".join(w.word for w in words))
step("Besten Abschnitt wählen")
window = pick_window(words)
if window is None:
    if total < MIN_S:
        print(f"Es gibt nur {total:.1f} s Sprache – das wird komplett verwendet (ab 3 s klappt es meist).")
    else:
        print("Kein Abschnitt passt sauber – nehme die ersten Wörter bis maximal 12 s.")
    j = max(k for k in range(len(words)) if words[k].end - words[0].start <= MAX_S) if total >= MIN_S else len(words) - 1
    window = (0, j)
cut(vocals, words, *window)
ensure_models()
