#!/usr/bin/env bash
# Referenz für Qwen-Stimmklonen vorbereiten – läuft auf dem eigenen PC (Docker).
#
#   prep_ref.sh QUELLE NAME              Stimme freistellen + Abschrift mit Zeitstempeln
#   prep_ref.sh QUELLE NAME START DAUER  Abschnitt ausschneiden -> NAME.wav + NAME.txt
#
# QUELLE: YouTube-Adresse (lädt nur den Ton per yt-dlp) oder lokale Datei.
# Ergebnis in ~/qwen-ref/NAME. Modelle/Pakete werden in ~/qwen-ref/.cache
# zwischengespeichert, der zweite Lauf lädt nichts neu.
set -euo pipefail

if [ $# -ne 2 ] && [ $# -ne 4 ]; then
  sed -n '3,9p' "$0"; exit 1
fi
SRC="$1"; NAME="$2"; START="${3:-}"; DUR="${4:-}"
if ! [[ "$NAME" =~ ^[a-z0-9_-]+$ ]]; then
  echo "NAME bitte nur aus Kleinbuchstaben, Ziffern, - und _ (wird Dateiname)."; exit 1
fi

BASE="$HOME/qwen-ref"
WORK="$BASE/$NAME"
CACHE="$BASE/.cache"
mkdir -p "$WORK" "$CACHE/pip" "$CACHE/torch" "$CACHE/hf"

# Quelle holen (nur beim ersten Lauf)
if ! ls "$WORK"/source.* >/dev/null 2>&1; then
  if [[ "$SRC" =~ ^https?:// ]]; then
    command -v yt-dlp >/dev/null || { echo "yt-dlp fehlt: sudo pacman -S yt-dlp"; exit 1; }
    yt-dlp -f bestaudio -x --audio-format wav -o "$WORK/source.%(ext)s" "$SRC"
  else
    [ -f "$SRC" ] || { echo "Datei nicht gefunden: $SRC"; exit 1; }
    cp "$SRC" "$WORK/source.${SRC##*.}"
  fi
fi

curl -fsSL -o "$BASE/prep_ref.py" \
  https://raw.githubusercontent.com/K3113rKind/k3113rkind-twitch-tts/qwen3-test/tools/qwen3-test/prep_ref.py

# Torch 2.5.1 (CPU) bewusst fest: Demucs speichert über torchaudio.save, das in
# neueren torchaudio-Versionen umgebaut wurde.
docker run --rm \
  -e REF_NAME="$NAME" -e REF_START="$START" -e REF_DUR="$DUR" -e WHISPER_MODEL \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -e TORCH_HOME=/cache/torch -e HF_HOME=/cache/hf \
  -v "$WORK":/work -v "$BASE/prep_ref.py":/prep_ref.py:ro \
  -v "$CACHE":/cache -v "$CACHE/pip":/root/.cache/pip \
  python:3.12-slim bash -c '
    set -e
    apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg >/dev/null
    pip install -q --root-user-action=ignore torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    pip install -q --root-user-action=ignore demucs soundfile faster-whisper
    python /prep_ref.py
    chown -R "$HOST_UID:$HOST_GID" /work /cache
  '

if [ -n "$START" ]; then
  echo
  echo "Auf den Test-CT kopieren:"
  echo "  scp $WORK/$NAME.wav $WORK/$NAME.txt root@192.168.0.176:/root/qwen-test/ref/"
fi
