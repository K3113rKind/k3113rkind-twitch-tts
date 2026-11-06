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
    && python -c "import io, urllib.request, zipfile; z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip').read())); z.extract('deno', '/usr/local/bin')" \
    && chmod +x /usr/local/bin/deno && deno --version | head -1 \
    && python -c "import torch, demucs, faster_whisper; print('Torch nach Klon-Werkzeugen:', torch.__version__)"

COPY app ./app
COPY static ./static

# Stimmen und Einstellungen liegen bewusst außerhalb des Images, in
# gemounteten Ordnern (siehe docker-compose.yml).
ENV MODELS_DIR=/models \
    CONFIG_PATH=/config/config.json \
    HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
