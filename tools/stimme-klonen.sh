#!/usr/bin/env bash
# Eigene Stimme für die Qwen-Stimmen erzeugen – fragt nach Quelle und Namen
# und erledigt den Rest: YouTube-Ton laden, Musik entfernen, abschreiben,
# besten 8–12-s-Abschnitt wählen, Referenz ablegen, Qwen-Base-Modelle laden.
# Danach steht die Stimme im Dropdown ("NAME · eigene Stimme, Qwen …").
#
# Aufruf im Repo-Ordner:  bash tools/stimme-klonen.sh
# Pfade über .env: QWEN_REF_DIR (Referenzen), QWEN_HF_DIR (Modelle).
set -euo pipefail
cd "$(dirname "$0")/.."

# Pfade aus .env (mit Standardwerten wie im Test-CT) und dort festhalten
touch .env
grep -q '^QWEN_HF_DIR=' .env || echo "QWEN_HF_DIR=/root/qwen-test/hf" >> .env
grep -q '^QWEN_REF_DIR=' .env || echo "QWEN_REF_DIR=/root/qwen-test/ref" >> .env
HF_DIR=$(grep '^QWEN_HF_DIR=' .env | tail -1 | cut -d= -f2-)
REF_DIR=$(grep '^QWEN_REF_DIR=' .env | tail -1 | cut -d= -f2-)
WORK_DIR="$(dirname "$REF_DIR")/klon-arbeit"
CACHE_DIR="$(dirname "$REF_DIR")/klon-cache"
mkdir -p "$HF_DIR" "$REF_DIR" "$WORK_DIR" "$CACHE_DIR/pip" "$CACHE_DIR/torch"

read -rp "YouTube-Adresse oder Pfad zu einer Audio-/Videodatei: " QUELLE
[ -n "$QUELLE" ] || { echo "Keine Quelle angegeben."; exit 1; }
while true; do
  read -rp "Name der Stimme (klein, ohne Leerzeichen, z. B. werbestimme): " NAME
  [[ "$NAME" =~ ^[a-z0-9_-]+$ ]] && break
  echo "Bitte nur a-z, 0-9, - und _."
done
if [ -f "$REF_DIR/$NAME.wav" ]; then
  read -rp "Stimme '$NAME' gibt es schon – überschreiben? [j/N] " OK
  [[ "$OK" =~ ^[jJyY]$ ]] || exit 1
  rm -rf "${WORK_DIR:?}/$NAME"
fi

# Lokale Datei in den Container reichen
QUELL_MOUNT=()
if [[ ! "$QUELLE" =~ ^https?:// ]]; then
  QUELLE=$(realpath "$QUELLE")
  [ -f "$QUELLE" ] || { echo "Datei nicht gefunden: $QUELLE"; exit 1; }
  QUELL_MOUNT=(-v "$(dirname "$QUELLE")":/quelle:ro)
fi

# Torch 2.5.1 (CPU) bewusst fest: Demucs speichert über torchaudio.save, das
# in neueren torchaudio-Versionen umgebaut wurde.
docker run --rm -it \
  -e QUELLE="$QUELLE" -e NAME="$NAME" -e WHISPER_MODEL \
  -e HF_HOME=/hf -e TORCH_HOME=/cache/torch \
  -v "$HF_DIR":/hf -v "$REF_DIR":/ref -v "$WORK_DIR":/arbeit \
  -v "$CACHE_DIR":/cache -v "$CACHE_DIR/pip":/root/.cache/pip \
  -v "$PWD/tools/stimme_klonen.py":/stimme_klonen.py:ro \
  "${QUELL_MOUNT[@]}" \
  python:3.12-slim bash -c '
    set -e
    echo "== Werkzeuge installieren (beim ersten Mal einige Minuten)"
    apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg >/dev/null
    pip install -q --root-user-action=ignore torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    pip install -q --root-user-action=ignore -U demucs soundfile faster-whisper huggingface_hub "yt-dlp[default]"
    python /stimme_klonen.py
  '

echo
echo "== App neu starten, damit sie den Referenzordner sieht"
docker compose up -d --build
echo
echo "Fertig. In der Weboberfläche neu laden und '$NAME · eigene Stimme, Qwen …' wählen."
echo "Anhören der Referenz: $REF_DIR/$NAME.wav – Text prüfen/korrigieren: $REF_DIR/$NAME.txt"
