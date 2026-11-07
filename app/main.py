"""FastAPI-Anwendung: REST-API, Audio-Stream, WebSocket-Status, statische GUI."""

from __future__ import annotations

import asyncio
import hmac
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import remote
from .config import INTERNAL_KEYS, ConfigStore
from .player import Broadcaster, ChatSpeaker
from .stream import AudioStream
from .tts import TTS
from .twitch import TwitchChatClient
from .voices import DEFAULT_VOICE, available_voices, is_available

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="K3113rkind's Twitch TTS")

config = ConfigStore()
tts = TTS()
broadcaster = Broadcaster()
audio_stream = AudioStream(config)
speaker = ChatSpeaker(config, tts, broadcaster, audio_stream)
remote.REGISTRY = remote.RemoteVoices(config)

state: dict = {"twitch": None, "error": None, "loading": False}

# Serialisiert Start/Stopp/Kanalwechsel. Ohne Lock konnten zwei gleichzeitige
# Aufrufe (z. B. Desktop + iPhone, oder Doppelklick) beide die "läuft schon?"-
# Prüfung passieren, während die Stimme lädt. Dann entstanden zwei IRC-Clients;
# der erste wurde überschrieben, aber nie gestoppt und las den alten Kanal
# weiter – auch nach Stopp und Kanalwechsel.
control_lock = asyncio.Lock()


async def _ensure_loaded(voice: str) -> None:
    if state["loading"]:
        return
    state["loading"] = True
    state["error"] = None
    try:
        await asyncio.to_thread(tts.load, voice)
    except Exception as exc:
        state["error"] = str(exc)
        raise
    finally:
        state["loading"] = False


def _status() -> dict:
    twitch: TwitchChatClient | None = state["twitch"]
    return {
        "running": twitch is not None,
        "connected": bool(twitch and twitch.connected),
        "channel": twitch.channel if twitch else None,
        "loading": state["loading"],
        "error": state["error"],
        "voices": available_voices(),
        "queue_size": speaker.queue_size,
        "listeners": broadcaster.count,
        "stream_listeners": audio_stream.listener_count,
    }


def _public_config() -> dict:
    """Konfiguration für die GUI – ohne die Kopplung zur Administrativ-Instanz."""
    data = config.as_dict()
    for key in INTERNAL_KEYS:
        data.pop(key, None)
    return data


@app.get("/api/config")
async def get_config():
    return _public_config()


@app.put("/api/config")
async def put_config(changes: dict):
    try:
        config.update(changes)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"Ungültiger Wert: {exc}")
    return _public_config()


@app.post("/api/remote/sync")
async def remote_sync(request: Request, body: dict):
    """Die Administrativ-Instanz überträgt ihre freigegebenen Stimmen
    (Schlüssel: remote_key dieser Instanz). Leere Liste = keine mehr."""
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not token or not hmac.compare_digest(token, config.get("remote_key") or ""):
        raise HTTPException(401, "Ungültiger Schlüssel.")
    url = str(body.get("url") or "").strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "Adresse der Administrativ-Instanz fehlt.")
    voices = remote.clean_voices(body.get("voices"))
    config.set_internal({"remote_url": url, "remote_voices": voices})
    # Gewählte Stimme nicht mehr freigegeben -> zurück zur Standardstimme
    if remote.is_remote(config.get("voice")) and not is_available(config.get("voice")):
        config.update({"voice": DEFAULT_VOICE})
        speaker.skip()
    log.info("Freigegebene Stimmen übernommen: %d von %s", len(voices), url)
    await broadcaster.send({"type": "status", **_status()})
    return {"ok": True, "count": len(voices)}


@app.get("/api/status")
async def get_status():
    return _status()


@app.post("/api/start")
async def start():
    async with control_lock:
        return await _start_locked()


async def _stop_client_locked() -> None:
    client: TwitchChatClient | None = state["twitch"]
    state["twitch"] = None
    if client is not None:
        await client.stop()
    await speaker.stop()


async def _start_locked():
    if state["twitch"] is not None:
        raise HTTPException(409, "Läuft bereits.")
    channel = config.get("channel")
    if not channel:
        raise HTTPException(400, "Bitte zuerst einen Twitch-Kanal eintragen.")

    voice = config.get("voice")
    if not is_available(voice):
        usable = available_voices()
        if not usable:
            raise HTTPException(
                400, "Keine Stimmen installiert – bitte install.sh erneut ausführen."
            )
        voice = usable[0]["key"]

    try:
        await _ensure_loaded(voice)
    except Exception as exc:
        raise HTTPException(500, f"Stimme konnte nicht geladen werden: {exc}")

    speaker.start()
    client = TwitchChatClient(channel, speaker.on_message)
    client.start()
    state["twitch"] = client
    log.info("Gestartet für Kanal #%s", channel)
    return _status()


@app.post("/api/stop")
async def stop():
    async with control_lock:
        if state["twitch"] is None:
            raise HTTPException(409, "Läuft nicht.")
        await _stop_client_locked()
        log.info("Gestoppt")
        return _status()


@app.post("/api/apply/channel")
async def apply_channel():
    """Kanal übernehmen: läuft gerade eine Verbindung, wird sie auf den in
    der Konfiguration hinterlegten Kanal umgehängt (stoppen + neu starten).
    Läuft keine, wird einfach gestartet – der Knopf tut also immer das,
    was der Nutzer erwartet."""
    async with control_lock:
        channel = config.get("channel")
        if not channel:
            raise HTTPException(400, "Bitte zuerst einen Twitch-Kanal eintragen.")

        client: TwitchChatClient | None = state["twitch"]
        if client is not None:
            if client.channel == channel.lstrip("#").lower():
                return _status()  # nichts zu tun
            old = client.channel
            await _stop_client_locked()
            log.info("Kanalwechsel: #%s -> #%s", old, channel)

        return await _start_locked()


@app.post("/api/apply/voice")
async def apply_voice():
    """Stimme sofort übernehmen: vorladen (damit die erste Nachricht nicht
    verzögert kommt) und die gerade laufende Ausgabe abbrechen."""
    voice = config.get("voice")
    if not is_available(voice):
        raise HTTPException(400, "Diese Stimme ist nicht installiert.")
    try:
        await _ensure_loaded(voice)
    except Exception as exc:
        raise HTTPException(500, f"Stimme konnte nicht geladen werden: {exc}")
    speaker.skip()
    return _status()


@app.post("/api/skip")
async def skip():
    speaker.skip()
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    broadcaster.add(ws)
    try:
        await ws.send_json({"type": "status", **_status()})
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "skip":
                speaker.skip()
            elif kind == "ping":
                # Lebenszeichen der Browser-Seite, damit Reverse-Proxys die
                # Verbindung nicht wegen Untätigkeit kappen.
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.remove(ws)


STREAM_HEADERS = {
    # Live-Stream: nichts zwischenspeichern, nichts umkodieren/puffern lassen
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
    "X-Accel-Buffering": "no",
    "icy-name": "K3113rkind's Twitch TTS",
}


@app.api_route("/stream", methods=["GET", "HEAD"])
async def stream(request: Request):
    """Endloser MP3-Stream mit allen Ansagen (Webradio-Prinzip).

    Wird von der GUI und dem Overlay per <audio> abgespielt. Anders als
    WebAudio hält ein laufendes Media-Element auf dem iPhone auch im
    Hintergrund und bei gesperrtem Display die Wiedergabe am Leben.
    Range-Anfragen werden bewusst ignoriert (200 statt 206), wie bei
    Icecast – einen Live-Stream kann man nicht spulen."""
    if request.method == "HEAD":
        return Response(media_type="audio/mpeg", headers=STREAM_HEADERS)
    return StreamingResponse(
        audio_stream.listen(), media_type="audio/mpeg", headers=STREAM_HEADERS
    )


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/overlay")
async def overlay():
    """Schlanke Seite für OBS-Browserquellen: nur Anzeige und Ton, keine
    Bedienelemente. Mit ?text=0 wird auch die Textanzeige ausgeblendet."""
    return FileResponse(STATIC_DIR / "overlay.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
