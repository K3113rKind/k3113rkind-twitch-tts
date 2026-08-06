#!/usr/bin/env bash
# K3113rkind's Twitch TTS starten. Fenster offen lassen; Strg+C beendet das Programm.
set -uo pipefail
cd "$(dirname "$0")"

DOCKER="docker"
$DOCKER info >/dev/null 2>&1 || DOCKER="sudo docker"

cleanup() { echo; echo "Wird beendet …"; $DOCKER compose down >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

echo "=========================================="
echo "  K3113rkind's Twitch TTS läuft"
echo "  Im Browser öffnen: http://localhost:8380"
echo "  Beenden: Strg+C oder Fenster schließen"
echo "=========================================="
$DOCKER compose up
