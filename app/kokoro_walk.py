"""Eigenes Kokoro-Stimmpaket aus einer eigenen Stimme suchen (Random Walk).

Idee und Bewertung nach KVoiceWalk von RobViren (Apache-2.0,
github.com/RobViren/kvoicewalk), angepasst an das deutsche Kokoro
(kikiri-Victoria-Backbone) und an mehrere Zielaufnahmen:

  - Ziel: die Referenzaufnahme der eigenen Stimme (/ref/NAME.wav) plus –
    falls vorhanden – mit Qwen erzeugtes Trainingsmaterial
    (/config/training/NAME/*.wav). Aus allen wird ein gemittelter
    Stimm-Fingerabdruck (Resemblyzer) gebildet.
  - Start: die Kokoro-Stimmpakete, die am ähnlichsten klingen.
  - Schritt: Zufallsrauschen auf das beste Paket; besser bewertet -> behalten.
  - Bewertung: harmonisches Mittel aus Ähnlichkeit zum Ziel, Stabilität
    (zwei verschiedene Sätze klingen nach derselben Stimme) und Klangmerkmalen.

Jede Verbesserung wird sofort als /ref/kokoro/NAME.pt gespeichert – die
Stimme erscheint in der Weboberfläche ("NAME · eigene Stimme, Kokoro") und
wird während der Suche immer besser. Abbrechen jederzeit möglich, mit
--weiter geht es vom letzten Stand aus weiter.

  python -m app.kokoro_walk candy-lang --schritte 10000
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from .voices import KOKORO_CUSTOM_BACKBONE, models_dir, ref_dir

OTHER_TEXT = "Kannst du mir sagen, wie spät es gerade bei dir ist? Ich muss gleich los."
SR = 24000


def log(text: str) -> None:
    print(time.strftime("%H:%M:%S"), text, flush=True)


# ------------------------------------------------------------------ Modelle
def stock_voicepacks() -> list[Path]:
    """Alle Kokoro-Stimmpakete (54 vom Original + die lokalen deutschen) als
    Ausgangsmaterial. Die Originale werden einmalig in den HF-Cache geladen."""
    packs = sorted((models_dir() / "voices").glob("*.pt"))
    code = ("from huggingface_hub import snapshot_download; "
            "print(snapshot_download('hexgrad/Kokoro-82M', allow_patterns=['voices/*.pt']))")
    env = dict(os.environ, HF_HUB_OFFLINE="0")
    env.pop("HF_TOKEN", None)
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if res.returncode == 0:
        packs += sorted(Path(res.stdout.strip().splitlines()[-1], "voices").glob("*.pt"))
    else:
        log(f"Original-Stimmpakete nicht ladbar, nur lokale: {res.stderr.strip()[-200:]}")
    return packs


class Speaker:
    def __init__(self) -> None:
        import torch
        from kokoro import KModel, KPipeline

        warnings.filterwarnings("ignore")
        kmodel = KModel(repo_id="hexgrad/Kokoro-82M",
                        config=str(models_dir() / "backbone" / "config.json"),
                        model=str(models_dir() / "backbone" / f"{KOKORO_CUSTOM_BACKBONE}.pth")).eval()
        self.pipeline = KPipeline(lang_code="d", repo_id="hexgrad/Kokoro-82M", model=kmodel)
        self.torch = torch

    def say(self, text: str, voice) -> np.ndarray:
        voice = voice.detach().to(device="cpu", dtype=self.torch.float32)
        chunks = [a.detach().cpu().numpy() if hasattr(a, "detach") else a
                  for _, _, a in self.pipeline(text, voice=voice, speed=1.0)]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


# --------------------------------------------------------------- Bewertung
def features(audio: np.ndarray) -> dict[str, float]:
    """Klangmerkmale (gekürzte Fassung der KVoiceWalk-Merkmale)."""
    import librosa

    f: dict[str, float] = {}
    f["rms"] = float(np.sqrt(np.mean(audio ** 2)))
    f["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    for name, fn in (("centroid", librosa.feature.spectral_centroid),
                     ("bandwidth", librosa.feature.spectral_bandwidth),
                     ("rolloff", librosa.feature.spectral_rolloff)):
        v = fn(y=audio, sr=SR)[0]
        f[name + "_m"], f[name + "_s"] = float(np.mean(v)), float(np.std(v))
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13)
    for i, row in enumerate(mfcc):
        f[f"mfcc{i}_m"], f[f"mfcc{i}_s"] = float(np.mean(row)), float(np.std(row))
    flat = librosa.feature.spectral_flatness(y=audio)[0]
    f["flat_m"] = float(np.mean(flat))
    pitches, mags = librosa.piptrack(y=audio, sr=SR)
    p = pitches[mags.argmax(axis=0), np.arange(mags.shape[1])]
    p = p[p > 0]
    f["pitch_m"] = float(np.mean(p)) if len(p) else 0.0
    f["pitch_s"] = float(np.std(p)) if len(p) else 0.0
    return f


class Scorer:
    def __init__(self, targets: list[Path]) -> None:
        import soundfile as sf
        from resemblyzer import VoiceEncoder, preprocess_wav

        self.encoder = VoiceEncoder(device="cpu", verbose=False)
        self.prep = preprocess_wav
        embeds, feats = [], []
        for path in targets:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            embeds.append(self.encoder.embed_utterance(preprocess_wav(audio, source_sr=sr)))
            if sr == SR:
                feats.append(features(audio))
        e = np.mean(embeds, axis=0)
        self.target = e / np.linalg.norm(e)
        self.target_feats = {k: float(np.mean([f[k] for f in feats])) for k in feats[0]}

    def embed(self, audio: np.ndarray) -> np.ndarray:
        return self.encoder.embed_utterance(self.prep(audio, source_sr=SR))

    def target_sim(self, audio: np.ndarray) -> float:
        return float(np.inner(self.embed(audio), self.target))

    def score(self, audio: np.ndarray, audio2: np.ndarray, target_sim: float) -> dict:
        self_sim = float(np.inner(self.embed(audio), self.embed(audio2)))
        f = features(audio)
        penalty = sum(abs((f[k] - v) / v) for k, v in self.target_feats.items() if v)
        feature_sim = max(0.01, (100.0 - penalty) / 100.0)
        values = np.array([max(target_sim, 1e-3), max(self_sim, 1e-3), feature_sim])
        weights = np.array([0.48, 0.5, 0.02])
        score = weights.sum() / np.sum(weights / values) * 100.0
        return {"score": float(score), "target": target_sim, "self": self_sim, "feature": feature_sim}


# -------------------------------------------------------------------- Suche
def main() -> None:
    import soundfile as sf
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--schritte", type=int, default=10000)
    ap.add_argument("--weiter", action="store_true", help="vom gespeicherten Stand weitermachen")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("KOKORO_WALK_THREADS", "8")))
    ap.add_argument("--population", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    ref_wav = ref_dir() / f"{args.name}.wav"
    ref_txt = ref_dir() / f"{args.name}.txt"
    if not ref_wav.is_file() or not ref_txt.is_file():
        sys.exit(f"Eigene Stimme {args.name} nicht gefunden in {ref_dir()}")
    # Text der Referenz, auf ~2 Sätze gekürzt (jeder Schritt spricht ihn)
    target_text = ref_txt.read_text(encoding="utf-8").strip()
    if len(target_text) > 180:
        cut = max(target_text.rfind(c, 0, 180) for c in ".!?")
        target_text = target_text[:cut + 1] if cut > 40 else target_text[:180]

    training = Path(os.environ.get("CONFIG_PATH", "/config/config.json")).parent / "training" / args.name
    targets = [ref_wav] + sorted(training.glob("*.wav"))[:40]
    log(f"Ziel: {ref_wav.name} + {len(targets) - 1} Trainingsaufnahmen, {args.threads} Threads")

    out_dir = ref_dir() / "kokoro"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = out_dir / f"{args.name}.pt"

    speaker = Speaker()
    scorer = Scorer(targets)

    def evaluate(voice, min_sim: float = 0.0) -> dict:
        audio = speaker.say(target_text, voice)
        if len(audio) < SR // 2:
            return {"score": 0.0, "target": 0.0, "audio": audio}
        sim = scorer.target_sim(audio)
        if sim <= min_sim:
            return {"score": 0.0, "target": sim, "audio": audio}
        res = scorer.score(audio, speaker.say(OTHER_TEXT, voice), sim)
        res["audio"] = audio
        return res

    # Ausgangsmaterial bewerten
    packs = []
    for path in stock_voicepacks():
        try:
            packs.append((path.stem, torch.load(path, map_location="cpu", weights_only=True).float()))
        except Exception as exc:
            log(f"{path.name} übersprungen: {exc}")
    from collections import Counter

    shape = Counter(tuple(p.shape) for _, p in packs).most_common(1)[0][0]
    packs = [(n, p) for n, p in packs if tuple(p.shape) == shape]
    log(f"Bewerte {len(packs)} Kokoro-Stimmpakete als Ausgangspunkt …")
    rated = []
    for name, pack in packs:
        r = evaluate(pack)
        rated.append((r["score"], name, pack))
        log(f"  {name:<14} Ähnlichkeit {r['target']:.3f}  Wertung {r['score']:.1f}")
    rated.sort(key=lambda x: x[0], reverse=True)
    population = [p for _, _, p in rated[:args.population]]
    log("Beste Ausgangspunkte: " + ", ".join(n for _, n, _ in rated[:args.population]))
    stacked = torch.stack(population)
    std = stacked.std(dim=0)

    if args.weiter and out_pt.is_file():
        best_voice = torch.load(out_pt, map_location="cpu", weights_only=True).float()
        log(f"Mache mit {out_pt.name} weiter")
    else:
        best_voice = stacked.mean(dim=0)
    best = evaluate(best_voice)
    log(f"Start: Ähnlichkeit {best['target']:.3f}, Wertung {best['score']:.1f}")

    def save(voice, res) -> None:
        tmp = out_pt.with_suffix(".pt.tmp")
        torch.save(voice.clone(), tmp)
        os.replace(tmp, out_pt)
        sf.write(out_dir / f"{args.name}-probe.wav", res["audio"], SR)

    save(best_voice, best)
    t0 = time.time()
    for step in range(1, args.schritte + 1):
        diversity = random.uniform(0.01, 0.15)
        voice = best_voice + torch.randn_like(best_voice) * std * diversity
        res = evaluate(voice, best["target"] * 0.98)
        if res["score"] > best["score"]:
            best, best_voice = res, voice
            save(best_voice, best)
            log(f"Schritt {step}: Ähnlichkeit {best['target']:.3f}, Stabilität "
                f"{best.get('self', 0):.3f}, Wertung {best['score']:.2f}  -> gespeichert")
        if step % 100 == 0:
            rate = (time.time() - t0) / step
            log(f"Fortschritt {step}/{args.schritte}, {rate:.1f} s/Schritt, "
                f"noch ca. {(args.schritte - step) * rate / 3600:.1f} h, "
                f"beste Ähnlichkeit {best['target']:.3f}")
    log(f"Fertig: {out_pt} (Ähnlichkeit {best['target']:.3f})")


if __name__ == "__main__":
    main()
