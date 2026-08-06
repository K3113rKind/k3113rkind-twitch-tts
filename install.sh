#!/usr/bin/env bash
# =============================================================================
# K3113rkind's Twitch TTS – Installation
# =============================================================================
# Ein Aufruf, fertig: prüft Docker (installiert es auf Wunsch), lädt alle
# Stimmen herunter, startet das Programm und legt ein Desktop-Symbol an.
# =============================================================================
set -euo pipefail

BOLD=$(tput bold 2>/dev/null || true); RESET=$(tput sgr0 2>/dev/null || true)
say()   { echo "${BOLD}»${RESET} $*"; }
warn()  { echo "${BOLD}»${RESET} ⚠ $*"; }
abort() { echo "${BOLD}»${RESET} ✗ $*" >&2; exit 1; }

ask_yn() {
    local prompt="$1" default="${2:-j}" answer
    if [[ "$default" == "j" ]]; then prompt="$prompt [J/n] "; else prompt="$prompt [j/N] "; fi
    read -r -p "$prompt" answer </dev/tty || true
    answer="${answer:-$default}"
    [[ "${answer,,}" == "j" || "${answer,,}" == "y" ]]
}

cd "$(dirname "$0")"
[[ -f docker-compose.yml && -d app ]] || abort "Bitte im entpackten Projektordner ausführen."
PROJECT_DIR="$(pwd)"

echo
echo "${BOLD}K3113rkind's Twitch TTS – Installation${RESET}"
echo "Der Twitch-Chat wird vorgelesen. Das dauert beim ersten Mal einige Minuten."
echo

# ------------------------------------------------------- SteamOS abfangen
if grep -qi '^ID=steamos' /etc/os-release 2>/dev/null; then
    abort "SteamOS wird hier nicht unterstützt (Systemordner sind schreibgeschützt)."
fi

# ------------------------------------------------------------ Docker
DOCKER="docker"
if ! command -v docker >/dev/null 2>&1; then
    warn "Docker fehlt – das wird zum Ausführen gebraucht."
    if ask_yn "Docker jetzt installieren?"; then
        if command -v pacman >/dev/null 2>&1; then
            sudo pacman -S --needed --noconfirm docker docker-compose
        else
            curl -fsSL https://get.docker.com | sudo sh
        fi
        sudo systemctl enable --now docker
        if ! groups "$USER" | grep -qw docker; then
            sudo usermod -aG docker "$USER"
            warn "Nach dem nächsten Ab-/Anmelden läuft Docker ohne sudo."
        fi
    else
        abort "Ohne Docker geht es leider nicht weiter."
    fi
fi
$DOCKER info >/dev/null 2>&1 || DOCKER="sudo docker"
$DOCKER info >/dev/null 2>&1 || abort "Docker läuft nicht. Bitte Rechner neu starten und nochmal versuchen."
$DOCKER compose version >/dev/null 2>&1 || abort "Docker-Compose fehlt. Bitte 'docker-compose-plugin' installieren."
say "Docker ist bereit."

# ------------------------------------------------------------ Stimmen
mkdir -p models/backbone models/voices config

HF="https://huggingface.co"
fetch() {  # fetch <url> <ziel> <anzeigename>
    local url="$1" dest="$2" name="$3"
    if [[ -f "$dest" ]]; then
        say "  $name – schon vorhanden"
        return 0
    fi
    say "  $name wird geladen …"
    curl -fL --progress-bar -o "$dest.part" "$url" && mv "$dest.part" "$dest"
}

say "Stimmen werden heruntergeladen (rund 1 GB, einmalig):"
fetch "$HF/hexgrad/Kokoro-82M/resolve/main/config.json" \
      models/backbone/config.json "Grundeinstellungen"
fetch "$HF/kikiri-tts/kikiri-german-victoria/resolve/main/kikiri_german_victoria_ep10.pth" \
      models/backbone/kikiri_german_victoria_ep10.pth "Victoria (deutsch, weiblich)"
fetch "$HF/kikiri-tts/kikiri-german-victoria/resolve/main/voices/victoria.pt" \
      models/voices/df_victoria.pt "Victoria – Stimmdaten"
fetch "$HF/kikiri-tts/kikiri-german-martin/resolve/main/kikiri_german_martin_ep10.pth" \
      models/backbone/kikiri_german_martin_ep10.pth "Martin (deutsch, männlich)"
fetch "$HF/kikiri-tts/kikiri-german-martin/resolve/main/voices/martin.pt" \
      models/voices/dm_martin.pt "Martin – Stimmdaten"
fetch "$HF/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth" \
      models/backbone/kokoro-v1_0.pth "Englische Stimmen"
fetch "$HF/hexgrad/Kokoro-82M/resolve/main/voices/af_heart.pt" \
      models/voices/af_heart.pt "Heart (englisch, weiblich)"
fetch "$HF/hexgrad/Kokoro-82M/resolve/main/voices/am_michael.pt" \
      models/voices/am_michael.pt "Michael (englisch, männlich)"
say "Alle Stimmen vorhanden."

# ------------------------------------------------------------ Bauen
echo
say "Programm wird vorbereitet (das dauert beim ersten Mal einige Minuten) …"
$DOCKER compose up -d --build
say "Läuft."

# ------------------------------------------------------- Desktop-Symbol
if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && ask_yn "Symbol auf dem Desktop anlegen?"; then
    APPS_DIR="$HOME/.local/share/applications"
    mkdir -p "$APPS_DIR"
    cat > "$APPS_DIR/k3113rkind-twitch-tts.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=K3113rkind's Twitch TTS
Comment=Liest den Twitch-Chat vor
Exec=$PROJECT_DIR/start.sh
Path=$PROJECT_DIR
Terminal=true
Icon=audio-speakers
Categories=AudioVideo;Audio;
EOF
    chmod +x "$APPS_DIR/k3113rkind-twitch-tts.desktop"
    for d in "$HOME/Desktop" "$HOME/Schreibtisch"; do
        [[ -d "$d" ]] && cp "$APPS_DIR/k3113rkind-twitch-tts.desktop" "$d/" && chmod +x "$d/k3113rkind-twitch-tts.desktop"
    done
    say "Symbol angelegt."
fi

echo
echo "${BOLD}Fertig!${RESET}"
echo "Im Browser öffnen:  ${BOLD}http://localhost:8380${RESET}"
echo "Dort Twitch-Kanal eintragen, Stimme wählen, auf 'Vorlesen starten' klicken."
echo
command -v xdg-open >/dev/null 2>&1 && ask_yn "Jetzt im Browser öffnen?" && xdg-open http://localhost:8380 >/dev/null 2>&1 || true
