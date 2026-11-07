"""Eigene Stimme aus Video-Link oder Audiodatei erzeugen (Qwen-Stimmklonen).

Läuft als eigener Prozess (python -m app.voice_clone …), gestartet von der
Weboberfläche: Speicher von Demucs/Whisper ist danach sofort wieder frei, und
ein Absturz trifft nicht die App. Ablauf:

  1. Ton holen: YouTube-/Twitch-Link per yt-dlp (mit Deno), oder Datei
  2. Stimme von Musik trennen (BS-RoFormer, Rückfall Demucs htdemucs; CPU)
  3. Mit Wort-Zeitstempeln abschreiben (faster-whisper, CPU)
  4. Besten Abschnitt der gewünschten Länge wählen (Satzgrenzen, keine langen
     Pausen, sicher erkannte Wörter)
  5. NAME.wav (24 kHz mono) + NAME.txt in den Referenzordner
  6. Qwen-Base-Modelle in den HF-Cache laden, falls sie fehlen

Fortschritt geht zeilenweise als "SCHRITT: …" auf stdout, das Ergebnis als
letzte Zeile "ERGEBNIS {json}".
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .voices import QWEN_BASE_MODELS, ref_dir

# large-v3: deutlich genauere Abschrift. Qwen klont aus Ton UND Text der
# Referenz – jedes falsch erkannte Wort verschlechtert den Klon.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
CANDIDATES = 3  # so viele Ausschnitte zur Auswahl
PAD_S = 0.15  # Luft vor/nach dem Abschnitt


def step(text: str) -> None:
    print(f"SCHRITT: {text}", flush=True)


def run(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


# ------------------------------------------------------------------ 1. Ton
def fetch(source: str, work: Path) -> Path:
    if source.startswith(("http://", "https://")):
        step("Ton aus dem Video laden (yt-dlp) …")
        run("yt-dlp", "--no-playlist", "--quiet", "--no-warnings", "-f", "bestaudio/best",
            "-x", "--audio-format", "wav", "-o", str(work / "quelle.%(ext)s"), source)
        found = sorted(work.glob("quelle.*"))
        if not found:
            raise RuntimeError("yt-dlp hat keine Audiodatei geliefert.")
        return found[0]
    path = Path(source)
    if not path.is_file():
        raise RuntimeError(f"Datei nicht gefunden: {source}")
    return path


def to_wav(src: Path, dst: Path, rate: int, channels: int) -> None:
    run("ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", str(channels),
        "-ar", str(rate), str(dst))


# ------------------------------------------------------------ 2. Trennen
SEPARATOR_PYTHON = os.environ.get("SEPARATOR_PYTHON", "/opt/separator/bin/python")
SEPARATOR_MODELS = os.environ.get("SEPARATOR_MODELS", "/hf/audio-separator")


def separate(source: Path, work: Path) -> Path:
    """Stimme freistellen. Bevorzugt BS-RoFormer (eigene venv, siehe
    app/separate_worker.py), Rückfall auf Demucs htdemucs, falls die venv
    fehlt oder das Modell nicht geladen werden kann."""
    stereo = work / "stereo.wav"
    to_wav(source, stereo, 44100, 2)
    vocals = work / "stimme_stereo.wav"
    if Path(SEPARATOR_PYTHON).is_file():
        step("Stimme von Musik trennen (BS-RoFormer, beim ersten Mal Modell-Download) …")
        os.makedirs(SEPARATOR_MODELS, exist_ok=True)
        worker = Path(__file__).with_name("separate_worker.py")
        res = subprocess.run([SEPARATOR_PYTHON, str(worker), str(stereo), str(vocals),
                              SEPARATOR_MODELS], capture_output=True, text=True)
        if res.returncode == 0 and vocals.is_file():
            for line in res.stderr.splitlines():
                if line.startswith(("Enthallt", "Enthallung übersprungen")):
                    step(line.strip()[:200])
            out = work / "stimme_komplett.wav"
            to_wav(vocals, out, 24000, 1)
            return out
        print("BS-RoFormer fehlgeschlagen, nehme Demucs:", res.stderr.strip()[-600:], flush=True)
    return separate_demucs(stereo, work)


def separate_demucs(stereo: Path, work: Path) -> Path:
    """Demucs über die Python-API; Ein-/Ausgabe per ffmpeg/soundfile statt
    torchaudio (dessen Speicherfunktionen wurden in neuen Versionen umgebaut)."""
    step("Stimme von Musik trennen (Demucs) …")
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    audio, _ = sf.read(stereo, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.T.copy())
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    model = get_model("htdemucs")
    model.eval()
    with torch.no_grad():
        sources = apply_model(model, ((wav - mean) / std)[None], device="cpu",
                              split=True, overlap=0.25, progress=False)[0]
    vocals = (sources[model.sources.index("vocals")] * std + mean).numpy()
    sf.write(work / "stimme_stereo.wav", vocals.T, 44100)
    out = work / "stimme_komplett.wav"
    to_wav(work / "stimme_stereo.wav", out, 24000, 1)
    del model, sources
    return out


# ---------------------------------------------------------- 3. Abschrift
def transcribe(path: Path, work: Path):
    step(f"Abschreiben (Whisper {WHISPER_MODEL}, beim ersten Mal Download) …")
    import soundfile as sf
    from faster_whisper import WhisperModel

    w16 = work / "whisper.wav"
    to_wav(path, w16, 16000, 1)
    audio, _ = sf.read(w16, dtype="float32")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio, language="de", beam_size=5, word_timestamps=True)
    words = [w for s in segments for w in (s.words or [])]
    for w in words:
        w.word = w.word.strip()
    return [w for w in words if w.word]


# ------------------------------------------------- 4. Abschnitt auswählen
def ends_sentence(word: str) -> bool:
    return word.endswith((".", "!", "?", "…"))


def pick_windows(words, ideal: float, count: int = CANDIDATES) -> list[tuple[int, int]]:
    """Die besten, sich nicht überschneidenden Abschnitte (bester zuerst).
    Bewertung wie bisher: Satzgrenzen, Länge nahe am Ziel, keine langen
    Pausen, sicher erkannte Wörter."""
    lo, hi = max(ideal - 2, 3.0), ideal + 2
    scored = []
    for i in range(len(words)):
        for j in range(i, len(words)):
            dur = words[j].end - words[i].start
            if dur > hi:
                break
            if dur < lo:
                continue
            seq = words[i:j + 1]
            gaps = [b.start - a.end for a, b in zip(seq, seq[1:])]
            score = 3.0 if i == 0 or ends_sentence(words[i - 1].word) else 0.0
            score += 3 if ends_sentence(words[j].word) else (1 if words[j].word.endswith(",") else 0)
            score -= abs(dur - ideal) * 0.5
            score -= sum(max(0.0, g - 0.6) for g in gaps) * 2
            score += 4 * (sum(w.probability for w in seq) / len(seq))
            scored.append((score, i, j))
    scored.sort(reverse=True)
    picked: list[tuple[int, int]] = []
    for _score, i, j in scored:
        a, b = words[i].start, words[j].end
        if all(b <= words[pi].start or a >= words[pj].end for pi, pj in picked):
            picked.append((i, j))
            if len(picked) >= count:
                break
    if not picked:  # zu wenig Sprache oder keine passende Länge
        j = max((k for k in range(len(words)) if words[k].end - words[0].start <= hi), default=len(words) - 1)
        picked = [(0, j)]
    return picked


def candidates_dir(name: str) -> Path:
    return ref_dir() / ".kandidaten" / name


def cut(vocals: Path, words, i: int, j: int, dst: Path) -> dict:
    """Abschnitt i..j als dst.wav (24 kHz mono, Lautheit angeglichen) + dst.txt."""
    prev_end = words[i - 1].end if i > 0 else 0.0
    next_start = words[j + 1].start if j + 1 < len(words) else words[j].end + PAD_S
    start = max(prev_end, words[i].start - PAD_S, 0.0)
    end = min(next_start, words[j].end + PAD_S)
    dur = end - start
    text = " ".join(w.word for w in words[i:j + 1])

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_wav = dst.parent / f".{dst.name}.wav.tmp"
    fade = f"afade=t=in:d=0.03,afade=t=out:st={max(dur - 0.05, 0):.3f}:d=0.05,loudnorm=I=-20:TP=-2"
    run("ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-i", str(vocals), "-af", fade, "-ac", "1", "-ar", "24000", "-f", "wav", str(tmp_wav))
    dst.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    os.replace(tmp_wav, dst.with_suffix(".wav"))  # erst jetzt sichtbar -> nie halbfertig
    return {"start": round(start, 1), "end": round(end, 1), "seconds": round(dur, 1), "text": text}


def cut_candidates(vocals: Path, words, windows, name: str) -> dict:
    """Alle Kandidaten nach .kandidaten/NAME/1..n, Kandidat 1 (der beste)
    zusätzlich als NAME.wav/.txt – die Stimme ist also sofort nutzbar und
    lässt sich in der Oberfläche auf einen anderen Kandidaten umstellen."""
    folder = candidates_dir(name)
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)
    cands = []
    for n, (i, j) in enumerate(windows, 1):
        info = cut(vocals, words, i, j, folder / str(n))
        cands.append({"idx": n, **info})
    best = folder / "1"
    tmp = ref_dir() / f".{name}.wav.tmp"
    shutil.copyfile(best.with_suffix(".wav"), tmp)
    shutil.copyfile(best.with_suffix(".txt"), ref_dir() / f"{name}.txt")
    os.replace(tmp, ref_dir() / f"{name}.wav")
    (folder / "gewaehlt").write_text("1\n", encoding="utf-8")
    return {"name": name, **cands[0], "candidates": cands, "chosen": 1}


# ---------------------------------------------------- 6. Base-Modelle
def ensure_models() -> None:
    from huggingface_hub import snapshot_download

    for repo in QWEN_BASE_MODELS.values():
        step(f"Modell {repo.split('/')[-1]} prüfen (beim ersten Mal ~2–4 GB Download) …")
        snapshot_download(repo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quelle", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sekunden", type=float, default=20)
    ap.add_argument("--arbeit", required=True)
    args = ap.parse_args()

    work = Path(args.arbeit)
    work.mkdir(parents=True, exist_ok=True)
    try:
        source = fetch(args.quelle, work)
        vocals = separate(source, work)
        words = transcribe(vocals, work)
        if not words:
            raise RuntimeError("Whisper hat keine Sprache gefunden – ist in der Aufnahme jemand zu hören?")
        step("Beste Abschnitte wählen …")
        result = cut_candidates(vocals, words, pick_windows(words, args.sekunden), args.name)
        ensure_models()
        print("ERGEBNIS " + json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
