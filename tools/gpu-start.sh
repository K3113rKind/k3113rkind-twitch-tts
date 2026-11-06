#!/usr/bin/env bash
# Twitch TTS mit AMD-GPU starten (eigener PC) – wird vom Desktop-Starter aufgerufen.
#
#   1. auf Updates prüfen und ggf. einspielen (git pull + Neubau)
#   2. Container starten, Weboberfläche im Browser öffnen
#   3. Log im Terminal anzeigen
#   Terminal schließen oder Strg+C -> Container wird gestoppt (Grafikspeicher frei)
set -uo pipefail
cd "$(dirname "$0")/.."

COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.gpu-amd.yml)
docker info >/dev/null 2>&1 || COMPOSE=(sudo "${COMPOSE[@]}")
URL="http://localhost:8380"

stopping=0
stop_container() {
  [ "$stopping" = 1 ] && return
  stopping=1
  trap '' HUP INT TERM
  # Beim Schließen des Fensters gibt es kein Terminal mehr – Ausgabe verwerfen
  { echo; echo "== Wird beendet …"; } 2>/dev/null
  "${COMPOSE[@]}" stop -t 5 >/dev/null 2>&1
}
trap 'stop_container; exit 0' HUP INT TERM
trap stop_container EXIT

echo "=============================================="
echo "  K3113rkind's Twitch TTS (GPU)"
echo "  Fenster schließen oder Strg+C = beenden"
echo "=============================================="

# ---------------------------------------------------------------- Updates
BUILD=()
echo "== Prüfe auf Updates …"
if git fetch -q origin 2>/dev/null; then
  LOCAL=$(git rev-parse HEAD)
  REMOTE=$(git rev-parse '@{u}' 2>/dev/null || echo "$LOCAL")
  if [ "$LOCAL" != "$REMOTE" ]; then
    echo "== Update gefunden – wird eingespielt:"
    git log --oneline "$LOCAL..$REMOTE" | sed 's/^/   /'
    if git pull -q --ff-only; then
      BUILD=(--build)
    else
      echo "   ⚠ Update nicht möglich (lokale Änderungen?) – starte die vorhandene Version."
    fi
  else
    echo "   Aktuell."
  fi
else
  echo "   ⚠ GitHub nicht erreichbar – starte die vorhandene Version."
fi
# Image fehlt (erster Start) -> bauen
docker image inspect k3113rkind-twitch-tts:rocm >/dev/null 2>&1 || BUILD=(--build)

# ----------------------------------------------------------------- Starten
echo "== Starte Container …"
if ! "${COMPOSE[@]}" up -d "${BUILD[@]}"; then
  echo "⚠ Start fehlgeschlagen. Fenster schließt mit Enter."
  read -r
  exit 1
fi

# Browser öffnen, sobald die Oberfläche antwortet (max. 60 s)
( for _ in $(seq 60); do
    if curl -fs -o /dev/null "$URL"; then xdg-open "$URL" >/dev/null 2>&1; exit 0; fi
    sleep 1
  done ) &

echo "== Läuft: $URL"
echo "----------------------------------------------"
# Log anzeigen; im Hintergrund + wait, damit Schließen/Strg+C sofort greift
"${COMPOSE[@]}" logs -f --tail 20 &
wait $!
