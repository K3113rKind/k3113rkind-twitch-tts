"""Messtest: Mehrere Chat-Nachrichten gleichzeitig mit Qwen erzeugen (Batching).

Läuft im laufenden Container, ohne Neubau (Skript kommt über stdin):

  docker compose exec -T twitch-tts python - candy-lang 1.7b 1,2,4,8 < tools/qwen-batch-test.py

Argumente: eigene Stimme, Modellgröße (0.6b/1.7b), Batch-Größen.
Gibt pro Batch-Größe Rechenzeit, erzeugtes Audio und RTF aus und legt die
Ergebnisse der größten Batch-Größe zum Anhören in den config-Ordner
(bench-batch-*.wav) – Batching könnte die Qualität verändern.
"""

import logging
import sys
import time

import numpy as np
import soundfile as sf
import torch

from app.tts import (QWEN_REF_DECODE_S, QWEN_SEED, QWEN_STEPS_PER_SECOND, QwenEngine,
                     _qwen_text, _qwen_token_budget)
from app.voices import QWEN_BASE_MODELS, ref_dir

logging.basicConfig(level=logging.WARNING)

name = sys.argv[1] if len(sys.argv) > 1 else "candy-lang"
size = sys.argv[2] if len(sys.argv) > 2 else "1.7b"
sizes = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2,4,8").split(",")]

# Typische Chat-Nachrichten, ähnlich lang
TEXTS = [
    "Moin Mouz, moin Chat.",
    "Die warn echt sehr spendabel heute.",
    "Was für ein krasser Clutch, unglaublich!",
    "Hast du schon die neue Map ausprobiert?",
    "Ich muss gleich los, bis später Leute.",
    "Spiel doch mal wieder mit dem Sniper.",
    "Das war knapp, fast hätte es nicht gereicht.",
    "Grüße aus dem Schwarzwald an alle hier.",
]

wav = ref_dir() / f"{name}.wav"
txt = ref_dir() / f"{name}.txt"
if not wav.is_file() or not txt.is_file():
    sys.exit(f"Eigene Stimme {name} nicht gefunden in {ref_dir()}")

print(f"Lade Qwen {size} Base, Stimme {name}, Torch-Threads {torch.get_num_threads()} …", flush=True)
eng = QwenEngine()
eng.load(QWEN_BASE_MODELS[size])
prompt = eng._clone_prompt(str(wav), str(txt))
ref_code = getattr(prompt[0], "ref_code", None)
trim = 0
if QWEN_REF_DECODE_S > 0 and ref_code is not None:
    trim = max(0, int(ref_code.shape[0]) - int(QWEN_REF_DECODE_S * QWEN_STEPS_PER_SECOND))


def run(texts):
    eng._trim_codes = trim
    try:
        torch.manual_seed(QWEN_SEED)
        t0 = time.perf_counter()
        wavs, sr = eng._model.generate_voice_clone(
            text=[_qwen_text(t) for t in texts],
            language=["German"] * len(texts),
            voice_clone_prompt=list(prompt) * len(texts),
            max_new_tokens=max(_qwen_token_budget(t) for t in texts),
            temperature=0.9,
        )
        took = time.perf_counter() - t0
    finally:
        eng._trim_codes = 0
    return took, wavs, int(sr)


print("Aufwärmen …", flush=True)
run(TEXTS[:1])

print()
print(f"{'Anzahl':>6} | {'Rechenzeit':>10} | {'Audio gesamt':>12} | {'RTF gesamt':>10} | {'je Nachricht':>12}")
print("-" * 64)
base = None
last = None
for n in sizes:
    texts = [TEXTS[i % len(TEXTS)] for i in range(n)]
    took, wavs, sr = run(texts)
    # Stille der Referenz-Auffüllung zählt nicht als Audio: nur den hörbaren Teil messen
    audio = sum(len(np.trim_zeros(np.asarray(w, dtype=np.float32), "f")) for w in wavs) / sr
    rtf = took / audio if audio else 0.0
    base = base or rtf
    print(f"{n:>6} | {took:>8.1f} s | {audio:>10.1f} s | {rtf:>10.2f} | {took / n:>10.1f} s", flush=True)
    last = (n, wavs, sr)

print()
print(f"RTF unter 1 = schneller als Echtzeit. Gewinn durch Batching: "
      f"{base / rtf:.1f}x (Batch {last[0]} gegenüber 1)" if rtf else "")

n, wavs, sr = last
for i, w in enumerate(wavs, 1):
    out = f"/config/bench-batch{n}-{i}.wav"
    sf.write(out, np.trim_zeros(np.asarray(w, dtype=np.float32), "f"), sr)
print(f"Zum Anhören: config/bench-batch{n}-1..{n}.wav im Projektordner")
