FROM python:3.12-slim

# espeak-ng: Lautschrift-Umwandlung; libsndfile1: WAV-Erzeugung
# git: kokoro/misaki werden aus Git-Repos installiert (siehe requirements.txt)
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

# Torch zuerst als reines CPU-Wheel, damit requirements.txt (kokoro -> torch)
# nicht die CUDA-Variante nachzieht.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static

# Stimmen und Einstellungen liegen bewusst außerhalb des Images, in
# gemounteten Ordnern (siehe docker-compose.yml).
ENV MODELS_DIR=/models \
    CONFIG_PATH=/config/config.json \
    HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
