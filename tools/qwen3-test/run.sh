#!/usr/bin/env bash
# Qwen3-TTS-Test im Wegwerf-Container – unabhängig vom Twitch-TTS-Container.
# Alles landet unter /root/qwen-test (Modelle, Pip-Cache, WAV-Ausgabe), ein
# erneuter Lauf lädt nichts doppelt. Aufräumen: rm -rf /root/qwen-test
set -euo pipefail

WORK=/root/qwen-test
THREADS="${THREADS:-16}"
mkdir -p "$WORK/out" "$WORK/hf" "$WORK/pip"
curl -fsSL -o "$WORK/bench.py" \
  https://raw.githubusercontent.com/K3113rKind/k3113rkind-twitch-tts/qwen3-test/tools/qwen3-test/bench.py

# Torch/Torchaudio zuerst als reine CPU-Wheels (sonst zieht qwen-tts die
# mehrere GB großen CUDA-Varianten). sox wird von qwen-tts benötigt.
docker run --rm -it --name qwen3-test \
  -e THREADS="$THREADS" -e SPEAKERS -e MODELS \
  -v "$WORK":/work -v "$WORK/hf":/root/.cache/huggingface -v "$WORK/pip":/root/.cache/pip \
  python:3.12-slim bash -c '
    set -e
    apt-get update -qq && apt-get install -y -qq --no-install-recommends sox libsox-fmt-all libsndfile1 >/dev/null
    pip install -q --root-user-action=ignore torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -q --root-user-action=ignore qwen-tts --extra-index-url https://download.pytorch.org/whl/cpu
    python -c "import torch; assert not torch.version.cuda, \"CUDA-Torch erwischt\"; print(\"CPU-Torch ok\")"
    python /work/bench.py
  '

echo
echo "WAVs liegen in $WORK/out. Zum Anhören im Browser:"
echo "  python3 -m http.server 8090 -d $WORK/out   ->  http://$(hostname -I | awk '{print $1}'):8090"
