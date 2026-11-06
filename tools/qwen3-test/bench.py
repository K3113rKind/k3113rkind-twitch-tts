"""Qwen3-TTS-Benchmark auf CPU: deutsche Chat-Sätze, Zeit pro Satz, WAVs zum Anhören.

Läuft im Wegwerf-Container aus run.sh. Gibt pro Satz die Rechenzeit, die
Audiodauer und den Echtzeitfaktor (RTF) aus. RTF < 1 heißt: schneller als
Echtzeit, also für den Chat brauchbar.

Umgebungsvariablen:
  THREADS   CPU-Threads für Torch (Default 16)
  SPEAKERS  Kommagetrennte Preset-Stimmen (Default Serena,Vivian,Ryan,Aiden)
  MODELS    Kommagetrennte Modell-IDs (Default 0.6B und 1.7B CustomVoice)
"""

import os
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

THREADS = int(os.environ.get("THREADS", "16"))
SPEAKERS = os.environ.get("SPEAKERS", "Serena,Vivian,Ryan,Aiden").split(",")
MODELS = os.environ.get(
    "MODELS",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice,Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
).split(",")
OUT = Path("/work/out")

# Typische Chat-Nachrichten: kurz, mittel, lang (mit Username wie im Bot)
TEXTS = [
    "Peter: Hallo zusammen!",
    "Mia sagt: Was für ein Schuss, das war richtig knapp gerade.",
    "Jonas: Kannst du nachher noch eine Runde mit dem schweren Panzer "
    "spielen? Die letzte Schlacht auf der Wüstenkarte war echt spannend.",
]

torch.set_num_threads(THREADS)
OUT.mkdir(parents=True, exist_ok=True)
print(f"Torch {torch.__version__}, {THREADS} Threads, Stimmen: {', '.join(SPEAKERS)}\n")

results = []
for model_id in MODELS:
    short = model_id.split("-12Hz-")[-1]
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(model_id, device_map="cpu", dtype=torch.float32)
    print(f"== {short}: geladen in {time.perf_counter() - t0:.1f} s")

    # Aufwärmen: der erste Lauf ist immer langsamer und zählt nicht
    model.generate_custom_voice(text="Test.", language="German", speaker=SPEAKERS[0])

    for spk in SPEAKERS:
        for i, text in enumerate(TEXTS, 1):
            t0 = time.perf_counter()
            wavs, sr = model.generate_custom_voice(text=text, language="German", speaker=spk)
            took = time.perf_counter() - t0
            dur = len(wavs[0]) / sr
            rtf = took / dur if dur else float("inf")
            name = f"{short}_{spk}_{i}.wav"
            sf.write(OUT / name, wavs[0], sr)
            results.append((short, spk, i, took, dur, rtf))
            print(f"   {spk:<8} Satz {i}: {took:5.1f} s Rechenzeit für {dur:4.1f} s Audio  (RTF {rtf:.2f})  -> {name}")
    del model

print("\n== Zusammenfassung (Mittelwert RTF je Modell, < 1 = schneller als Echtzeit)")
for short in dict.fromkeys(r[0] for r in results):
    rtfs = [r[5] for r in results if r[0] == short]
    first = [r[3] for r in results if r[0] == short and r[2] == 1]
    print(f"   {short}: RTF {sum(rtfs) / len(rtfs):.2f}, kurzer Satz im Schnitt {sum(first) / len(first):.1f} s")
