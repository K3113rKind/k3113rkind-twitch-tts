#!/usr/bin/env bash
# Legt den Starter "Twitch TTS (GPU)" auf dem Desktop und im Startmenü an.
# Er ruft tools/gpu-start.sh in einem Terminal auf: Updates prüfen, starten,
# Log anzeigen – Terminal schließen beendet den Container.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
chmod +x "$PROJECT_DIR/tools/gpu-start.sh"

APPS_DIR="$HOME/.local/share/applications"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
mkdir -p "$APPS_DIR"
FILE="$APPS_DIR/k3113rkind-twitch-tts-gpu.desktop"

cat > "$FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Twitch TTS (GPU)
Comment=Updates prüfen, starten und Log anzeigen – Fenster schließen beendet
Exec=$PROJECT_DIR/tools/gpu-start.sh
Path=$PROJECT_DIR
Terminal=true
Icon=audio-speakers
Categories=AudioVideo;Audio;
EOF
chmod +x "$FILE"

if [ -d "$DESKTOP_DIR" ]; then
  cp "$FILE" "$DESKTOP_DIR/" && chmod +x "$DESKTOP_DIR/$(basename "$FILE")"
  echo "Starter angelegt: $DESKTOP_DIR/$(basename "$FILE") (und im Startmenü)"
else
  echo "Kein Desktop-Ordner gefunden – Starter nur im Startmenü: $FILE"
fi
