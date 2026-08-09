"""FastAPI-Anwendung: REST-API, WebSocket-Audio, statische GUI."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import ConfigStore
from .player import Broadcaster, ChatSpeaker
from .tts import TTS
from .twitch import TwitchChatClient
from .voices import available_voices, is_available

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="K3113rkind's Twitch TTS")

config = ConfigStore()
tts = TTS()
broadcaster = Broadcaster()
speaker = ChatSpeaker(config, tts, broadcaster)

state: dict = {"twitch": None, "error": None, "loading": False}


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
    }


@app.get("/api/config")
async def get_config():
    return config.as_dict()


@app.put("/api/config")
async def put_config(changes: dict):
    return config.update(changes)


@app.get("/api/status")
async def get_status():
    return _status()


@app.post("/api/start")
async def start():
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
    client: TwitchChatClient | None = state["twitch"]
    if client is None:
        raise HTTPException(409, "Läuft nicht.")
    await client.stop()
    state["twitch"] = None
    await speaker.stop()
    log.info("Gestoppt")
    return _status()


@app.post("/api/apply/channel")
async def apply_channel():
    """Kanal übernehmen: läuft gerade eine Verbindung, wird sie auf den in
    der Konfiguration hinterlegten Kanal umgehängt (stoppen + neu starten).
    Läuft keine, wird einfach gestartet – der Knopf tut also immer das,
    was der Nutzer erwartet."""
    channel = config.get("channel")
    if not channel:
        raise HTTPException(400, "Bitte zuerst einen Twitch-Kanal eintragen.")

    client: TwitchChatClient | None = state["twitch"]
    if client is not None:
        if client.channel == channel:
            return _status()  # nichts zu tun
        await client.stop()
        state["twitch"] = None
        await speaker.stop()
        log.info("Kanalwechsel: #%s -> #%s", client.channel, channel)

    return await start()


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
            if kind == "played":
                speaker.notify_played(int(msg.get("id", -1)))
            elif kind == "skip":
                speaker.skip()
            elif kind == "ping":
                # Lebenszeichen der Browser-Seite, damit Reverse-Proxys die
                # Verbindung nicht wegen Untätigkeit kappen.
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.remove(ws)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/overlay")
async def overlay():
    """Schlanke Seite für OBS-Browserquellen: nur Anzeige und Ton, keine
    Bedienelemente. Mit ?text=0 wird auch die Textanzeige ausgeblendet."""
    return FileResponse(STATIC_DIR / "overlay.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
