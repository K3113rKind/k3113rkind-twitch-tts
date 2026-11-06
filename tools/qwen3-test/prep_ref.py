"""Referenz für Qwen-Stimmklonen vorbereiten: Musik weg, abschreiben, ausschneiden.

Läuft im Wegwerf-Container aus prep_ref.sh. Arbeitsordner /work enthält
source.* (Video/Audio). Zwei Durchläufe:

  1. ohne REF_START/REF_DUR: Stimme freistellen (Demucs) und mit Zeitstempeln
     abschreiben (Whisper) – zum Aussuchen des besten Abschnitts.
  2. mit REF_START/REF_DUR: Abschnitt ausschneiden, NAME.wav + NAME.txt
     schreiben. Die Freistellung aus Lauf 1 wird wiederverwendet.
"""

import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/work")
NAME = os.environ["REF_NAME"]
START = os.environ.get("REF_START", "").strip()
DUR = os.environ.get("REF_DUR", "").strip()
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
VOCALS = WORK / "stimme_komplett.wav"


def run(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


def separate() -> None:
    if VOCALS.exists():
        print(f"Freigestellte Stimme vorhanden ({VOCALS.name}), überspringe Demucs.")
        return
    sources = sorted(WORK.glob("source.*"))
    if not sources:
        sys.exit("Keine Quelldatei (source.*) im Arbeitsordner gefunden.")
    print("== Stimme von Musik trennen (Demucs htdemucs) …")
    run("ffmpeg", "-loglevel", "error", "-y", "-i", str(sources[0]), "-ac", "2", "-ar", "44100", "/tmp/in.wav")
    run(sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems=vocals", "-o", "/tmp/sep", "/tmp/in.wav")
    run("ffmpeg", "-loglevel", "error", "-y", "-i", "/tmp/sep/htdemucs/in/vocals.wav",
        "-ac", "1", "-ar", "24000", str(VOCALS))


def transcribe(path: Path):
    from faster_whisper import WhisperModel

    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(path), language="de", beam_size=5)
    return list(segments)


def overview() -> None:
    print(f"\n== Abschrift mit Zeitstempeln (Whisper {WHISPER_MODEL})\n")
    for s in transcribe(VOCALS):
        print(f"  {s.start:6.1f} – {s.end:6.1f} s  ({s.end - s.start:4.1f} s)  {s.text.strip()}")
    print(
        "\nAnhören: stimme_komplett.wav im Arbeitsordner (Musik sollte weg sein).\n"
        "Jetzt einen Abschnitt von 8–12 s mit klarer Sprache wählen – am besten\n"
        "von Satzanfang bis Satzende, sonst passt die Abschrift nicht exakt.\n"
        "Dann das Skript erneut aufrufen, zusätzlich mit START und DAUER in Sekunden."
    )


def cut() -> None:
    start, dur = float(START.replace(",", ".")), float(DUR.replace(",", "."))
    out_wav = WORK / f"{NAME}.wav"
    fade = f"afade=t=in:d=0.05,afade=t=out:st={max(dur - 0.05, 0):.2f}:d=0.05"
    run("ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start}", "-t", f"{dur}", "-i", str(VOCALS),
        "-af", fade, "-ac", "1", "-ar", "24000", str(out_wav))
    text = " ".join(s.text.strip() for s in transcribe(out_wav)).strip()
    (WORK / f"{NAME}.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\n== {out_wav.name} ({dur:.1f} s) und {NAME}.txt geschrieben. Abschrift:\n\n  {text}\n")
    print("Bitte prüfen: Der Text muss wortgenau dem Gesagten entsprechen (sonst die .txt korrigieren).")


separate()
if START and DUR:
    cut()
else:
    overview()
