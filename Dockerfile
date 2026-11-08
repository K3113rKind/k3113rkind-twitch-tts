FROM python:3.12-slim

# espeak-ng: Lautschrift-Umwandlung; libsndfile1: WAV-Erzeugung
# git: kokoro/misaki werden aus Git-Repos installiert (siehe requirements.txt)
# libatomic1, libnuma1, libdrm*, libelf: werden von den ROCm-Torch-Wheels
# (GPU-Variante) vorausgesetzt, fehlen aber im schlanken Basis-Image; klein,
# deshalb für alle Varianten.
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng libsndfile1 git sox \
       libatomic1 libnuma1 libdrm2 libdrm-amdgpu1 libelf1t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

# Torch zuerst aus dem passenden Index, damit requirements.txt (kokoro -> torch)
# nicht die große CUDA-Variante von PyPI nachzieht. Standard: reine CPU.
# GPU-Varianten setzen TORCH_INDEX per Compose-Override, z. B.
#   AMD (ROCm):  https://download.pytorch.org/whl/rocm7.2   (docker-compose.gpu-amd.yml)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchaudio --index-url "$TORCH_INDEX" \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir qwen-tts --extra-index-url "$TORCH_INDEX" \
    && python -c "import torch; print('Torch', torch.__version__, 'HIP', torch.version.hip, 'CUDA', torch.version.cuda)" \
    && case "$TORCH_INDEX" in *cpu) python -c "import torch; assert not torch.version.cuda, 'CUDA-Torch im CPU-Image'";; esac

# GPU-Varianten: C-Compiler für torch.compile (Triton baut damit kleine
# Startroutinen). Eigene Schicht nach Torch, damit sie dessen Cache nicht
# ungültig macht; im CPU-Image entfällt sie.
RUN case "$TORCH_INDEX" in *cpu) ;; *) apt-get update \
      && apt-get install -y --no-install-recommends gcc libc6-dev \
      && rm -rf /var/lib/apt/lists/* ;; esac

# Eigene Stimmen aus der Weboberfläche erstellen (app/voice_clone.py):
# ffmpeg, Demucs (Musik entfernen), faster-whisper (Abschrift), yt-dlp mit
# Deno (YouTube verlangt seit Ende 2025 eine JavaScript-Laufzeit),
# python-multipart (Datei-Upload). Eigene Schichten nach Torch, damit dessen
# großer Download im Cache bleibt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir python-multipart demucs faster-whisper "yt-dlp[default]" \
       --extra-index-url "$TORCH_INDEX" \
    && python -c "import io, time, urllib.request, zipfile; url = 'https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip'; exec('for n in range(6):\n try:\n  data = urllib.request.urlopen(url, timeout=120).read(); break\n except Exception as e:\n  print(\'deno-Download fehlgeschlagen (%s), neuer Versuch in %d s\' % (e, 10 * (n + 1)), flush=True); time.sleep(10 * (n + 1))\nelse:\n raise SystemExit(\'deno nicht ladbar (GitHub)\')'); zipfile.ZipFile(io.BytesIO(data)).extract('deno', '/usr/local/bin')" \
    && chmod +x /usr/local/bin/deno && deno --version | head -1 \
    && python -c "import torch, demucs, faster_whisper; print('Torch nach Klon-Werkzeugen:', torch.__version__)"

# Pocket TTS (Kyutai): kleines CPU-Modell mit Deutsch und Stimmklonen.
# Eigene Schicht nach Torch; prüft, dass Torch nicht ausgetauscht wurde.
RUN TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)") \
    && pip install --no-cache-dir pocket-tts --extra-index-url "$TORCH_INDEX" \
    && python -c "import torch, sys; v=torch.__version__; print('Torch nach Pocket TTS:', v); sys.exit(v != '$TORCH_BEFORE')" \
    && python -c "import pocket_tts; print('pocket-tts', __import__('importlib.metadata').metadata.version('pocket-tts'))"

# Zusätzliche Test-Engines in eigener venv (NeuTTS, Chatterbox, VoxCPM2,
# OmniVoice) wurden im Nov 2026 wieder entfernt. CPU-Torch-Index für die
# Separator-venv:
ARG CPU_TORCH_INDEX=https://download.pytorch.org/whl/cpu

# Stimmtrennung für eigene Stimmen: BS-RoFormer über python-audio-separator
# (MIT) in eigener venv – bringt eigene Torch-/ONNX-Versionen mit. Das Modell
# lädt der erste Klon-Vorgang einmalig nach /hf/audio-separator.
# diffq (Abhängigkeit) wird aus dem Quellcode gebaut -> gcc nur für diese
# Schicht installieren und danach wieder entfernen (nicht in der GPU-Variante,
# dort braucht torch.compile gcc dauerhaft). librosa < 1 und audioread
# ausdrücklich: librosa 1.0 bringt audioread nicht mehr mit, audio-separator
# importiert es aber noch.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && python -m venv /opt/separator \
    && /opt/separator/bin/pip install --no-cache-dir torch --index-url "$CPU_TORCH_INDEX" \
    && /opt/separator/bin/pip install --no-cache-dir "audio-separator[cpu]" "librosa<1" audioread \
         --extra-index-url "$CPU_TORCH_INDEX" \
    && /opt/separator/bin/python -c "from audio_separator.separator import Separator; print('audio-separator ok')" \
    && case "$TORCH_INDEX" in *cpu) apt-get purge -y gcc libc6-dev && apt-get autoremove -y ;; esac \
    && rm -rf /var/lib/apt/lists/*

# Eigene Kokoro-Stimmpakete suchen (app/kokoro_walk.py): Resemblyzer als
# Stimm-Fingerabdruck. webrtcvad als fertiges Wheel (das Original müsste
# kompiliert werden), resemblyzer deshalb ohne Abhängigkeiten.
RUN pip install --no-cache-dir webrtcvad-wheels \
    && pip install --no-cache-dir --no-deps resemblyzer \
    && python -c "import resemblyzer, librosa; print('resemblyzer ok')"

COPY app ./app
COPY static ./static

# Stimmen und Einstellungen liegen bewusst außerhalb des Images, in
# gemounteten Ordnern (siehe docker-compose.yml).
ENV MODELS_DIR=/models \
    CONFIG_PATH=/config/config.json \
    HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
