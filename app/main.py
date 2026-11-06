"""FastAPI-Anwendung: REST-API, Audio-Stream, WebSocket-Status, statische GUI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import re
import shutil
import sys
import tempfile
from collections import deque
from pathlib import Path

from fastapi import (FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import ConfigStore
from .player import Broadcaster, ChatSpeaker
from .stream import AudioStream
from .tts import TTS
from .twitch import TwitchChatClient
from .voices import (DEFAULT_VOICE, available_voices, clone_keys, is_available, is_qwen,
                     ref_dir, sync_clone_voices)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


class _QuietPolling(logging.Filter):
    """Die Oberfläche fragt alle paar Sekunden /api/status ab – diese Zeilen
    verdecken im Terminal sonst alles Wichtige."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/status" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_QuietPolling())

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# Oberfläche immer frisch prüfen lassen (mit ETag kostet das nur ein 304).
# Ohne diesen Header schätzt der Browser die Gültigkeit anhand des
# Dateialters – nach einem Update zeigte er so tagelang die alte Seite.
NO_CACHE = {"Cache-Control": "no-cache"}


class FreshStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers.update(NO_CACHE)
        return response

app = FastAPI(title="K3113rkind's Twitch TTS")

config = ConfigStore()
tts = TTS()
broadcaster = Broadcaster()
audio_stream = AudioStream(config)
speaker = ChatSpeaker(config, tts, broadcaster, audio_stream)

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
        "warming": tts.warming,
        "error": state["error"],
        "voices": available_voices(config.get("voice_labels")),
        "queue_size": speaker.queue_size,
        "listeners": broadcaster.count,
        "stream_listeners": audio_stream.listener_count,
        "last_perf": speaker.last_perf,
    }


@app.get("/api/config")
async def get_config():
    return config.as_dict()


@app.put("/api/config")
async def put_config(changes: dict):
    try:
        return config.update(changes)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"Ungültiger Wert: {exc}")


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
    # Mehrere Qwen-Rechenprozesse: Modell gleich in allen laden, nicht erst
    # bei den ersten Nachrichten (je ~10 s Wartezeit)
    if tts.pool_active and is_qwen(voice) and not tts.warming:
        asyncio.create_task(asyncio.to_thread(tts.warmup, voice))
    return _status()


@app.post("/api/say")
async def say(body: dict):
    """Testtext vorlesen, ohne dass ein Twitch-Chat laufen muss. Nutzt die
    gerade eingestellte Stimme und Geschwindigkeit."""
    text = str(body.get("text", "")).strip()
    if not text:
        raise HTTPException(400, "Bitte einen Text eingeben.")
    if state["twitch"] is not None:
        raise HTTPException(409, "Testtext ist gesperrt, solange der Chat vorgelesen wird.")

    voice = config.get("voice")
    if not is_available(voice):
        usable = available_voices()
        if not usable:
            raise HTTPException(400, "Keine Stimmen installiert – bitte install.sh erneut ausführen.")
        voice = usable[0]["key"]
    try:
        await _ensure_loaded(voice)
    except Exception as exc:
        raise HTTPException(500, f"Stimme konnte nicht geladen werden: {exc}")

    await speaker.say_text(text)
    return {"ok": True, "queue_size": speaker.queue_size}


@app.post("/api/say/cancel")
async def say_cancel():
    """Testtexte abbrechen (laufenden und wartende); Twitch bleibt unberührt."""
    await speaker.cancel_test()
    return {"ok": True}


@app.on_event("startup")
async def warmup_on_start():
    """Gewählte Qwen-Stimme gleich nach dem Start laden und auf der GPU
    aufwärmen (QWEN_WARMUP=0 schaltet das ab)."""
    if os.environ.get("QWEN_WARMUP", "1") == "0":
        return
    asyncio.create_task(asyncio.to_thread(tts.warmup, config.get("voice")))


# ------------------------------------------------- eigene Stimme erzeugen
CLONE_NAME = re.compile(r"^[a-z0-9_-]{1,40}$")
UPLOAD_TYPES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4", ".webm", ".mkv"}
clone_job: dict = {"running": False, "name": None, "log": deque(maxlen=40),
                   "result": None, "error": None}


async def _run_clone(source: str, name: str, seconds: float, root: str) -> None:
    """Pipeline als eigener Prozess (app/voice_clone.py); Ausgabe -> Job-Log.
    root: temporärer Ordner (Upload + Arbeitsdateien), wird danach gelöscht."""
    work = str(Path(root) / "arbeit")
    env = {**os.environ, "HF_HUB_OFFLINE": "0", "TORCH_HOME": "/hf/torch",
           "PYTHONUNBUFFERED": "1"}
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "app.voice_clone", "--quelle", source, "--name", name,
            "--sekunden", str(seconds), "--arbeit", work,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env,
        )
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if not line:
                continue
            log.info("Klonen: %s", line)
            if line.startswith("ERGEBNIS "):
                clone_job["result"] = json.loads(line[len("ERGEBNIS "):])
            elif line.startswith("SCHRITT: "):
                clone_job["log"].append(line[len("SCHRITT: "):])
            else:
                clone_job["last_line"] = line[-300:]
        code = await proc.wait()
        if code != 0 or not clone_job["result"]:
            clone_job["error"] = clone_job.get("last_line") or f"Abbruch mit Code {code}"
        else:
            clone_job["log"].append("Fertig – die Stimme steht jetzt im Dropdown.")
            sync_clone_voices()
            await broadcaster.send({"type": "voices_changed"})
    except Exception as exc:
        clone_job["error"] = str(exc)
    finally:
        clone_job["running"] = False
        shutil.rmtree(root, ignore_errors=True)


@app.post("/api/clone")
async def clone_start(
    name: str = Form(...),
    seconds: float = Form(20),
    url: str = Form(""),
    overwrite: bool = Form(False),
    file: UploadFile | None = File(None),
):
    """Neue eigene Stimme aus Video-Link (YouTube/Twitch) oder Audiodatei."""
    name = name.strip().lower()
    if not CLONE_NAME.match(name):
        raise HTTPException(400, "Name bitte nur aus a–z, 0–9, - und _ (max. 40 Zeichen).")
    if clone_job["running"]:
        raise HTTPException(409, f"Es wird gerade schon „{clone_job['name']}“ erstellt.")
    if (ref_dir() / f"{name}.wav").exists() and not overwrite:
        raise HTTPException(409, "EXISTS")
    url = url.strip()
    has_file = file is not None and file.filename
    if bool(url) == bool(has_file):
        raise HTTPException(400, "Bitte entweder einen Video-Link oder eine Datei angeben.")
    if url and not re.match(r"^https?://", url):
        raise HTTPException(400, "Der Link muss mit http:// oder https:// beginnen.")
    seconds = min(max(float(seconds), 5.0), 30.0)

    work = tempfile.mkdtemp(prefix="klon-")
    source = url
    if has_file:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in UPLOAD_TYPES:
            raise HTTPException(400, "Bitte eine Audio- oder Videodatei (.wav, .mp3, …) hochladen.")
        source = str(Path(work) / f"upload{suffix}")
        with open(source, "wb") as fh:
            shutil.copyfileobj(file.file, fh)

    clone_job.update(running=True, name=name, result=None, error=None, last_line=None)
    clone_job["log"].clear()
    clone_job["log"].append("Gestartet …")
    asyncio.create_task(_run_clone(source, name, seconds, work))
    return _clone_status()


def _clone_status() -> dict:
    return {k: (list(v) if isinstance(v, deque) else v)
            for k, v in clone_job.items() if k != "last_line"}


@app.get("/api/clone")
async def clone_status():
    return _clone_status()


@app.put("/api/clone/text")
async def clone_text(body: dict):
    """Abschrift einer eigenen Stimme korrigieren (muss wortgenau stimmen)."""
    name = str(body.get("name", "")).strip().lower()
    text = str(body.get("text", "")).strip()
    if not CLONE_NAME.match(name) or not (ref_dir() / f"{name}.wav").exists():
        raise HTTPException(404, "Diese eigene Stimme gibt es nicht.")
    if not text:
        raise HTTPException(400, "Der Text darf nicht leer sein.")
    (ref_dir() / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
    if clone_job.get("result") and clone_job["result"].get("name") == name:
        clone_job["result"]["text"] = text
    return {"ok": True}


@app.get("/api/clone/voices")
async def clone_voices():
    """Eigene Stimmen im Referenzordner (Name, Größe, Text)."""
    folder = ref_dir()
    out = []
    if folder.is_dir():
        for wav in sorted(folder.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            size = wav.stat().st_size + (txt.stat().st_size if txt.exists() else 0)
            out.append({"name": wav.stem, "kb": round(size / 1024),
                        "text": txt.read_text(encoding="utf-8").strip() if txt.exists() else ""})
    return out


@app.delete("/api/clone/voices/{name}")
async def clone_delete(name: str):
    """Eigene Stimme löschen: Dateien, Umbenennungen, RAM-Cache. War sie
    ausgewählt, wird auf die Standardstimme umgeschaltet."""
    name = name.strip().lower()
    folder = ref_dir()
    if not CLONE_NAME.match(name) or not (folder / f"{name}.wav").exists():
        raise HTTPException(404, "Diese eigene Stimme gibt es nicht.")
    if clone_job.get("running") and clone_job.get("name") == name:
        raise HTTPException(409, "Diese Stimme wird gerade erstellt.")
    keys = clone_keys(name)
    changes = {}
    if config.get("voice") in keys:
        changes["voice"] = DEFAULT_VOICE
        speaker.skip()
    labels = dict(config.get("voice_labels") or {})
    if any(k in labels for k in keys):
        changes["voice_labels"] = {k: v for k, v in labels.items() if k not in keys}
    if changes:
        config.update(changes)
    for suffix in (".wav", ".txt"):
        (folder / f"{name}{suffix}").unlink(missing_ok=True)
    (folder / "kokoro" / f"{name}.pt").unlink(missing_ok=True)
    sync_clone_voices()
    tts.forget_clone(name)
    log.info("Eigene Stimme %s gelöscht", name)
    await broadcaster.send({"type": "voices_changed"})
    return {"ok": True, "voice": config.get("voice")}


training_job: dict = {"running": False, "name": None, "done": 0, "total": 0, "error": None}


@app.post("/api/training/{name}")
async def training_start(name: str, size: str = "0.6b"):
    """Trainingsmaterial für eine eigene Stimme mit Qwen erzeugen
    (/config/training/NAME/NNN.wav + .txt) – Grundlage für ein eigenes
    Kokoro-Stimmpaket (app/kokoro_walk.py). Nur bei gestopptem Chat."""
    from .training_texts import TRAINING_TEXTS

    name = name.strip().lower()
    voice = f"klon-{size}-{name}"
    sync_clone_voices()
    if not CLONE_NAME.match(name) or not is_available(voice):
        raise HTTPException(404, f"Eigene Stimme {name} ({size}) nicht verfügbar.")
    if state["twitch"] is not None:
        raise HTTPException(409, "Bitte erst „Vorlesen stoppen“.")
    if training_job["running"]:
        raise HTTPException(409, "Es läuft schon eine Erzeugung.")
    out = Path(os.environ.get("CONFIG_PATH", "/config/config.json")).parent / "training" / name
    out.mkdir(parents=True, exist_ok=True)
    training_job.update(running=True, name=name, done=0, total=len(TRAINING_TEXTS), error=None)
    temperature = config.get("qwen_temperature")

    async def one(i: int, text: str) -> None:
        import soundfile as sf

        result = await asyncio.to_thread(tts.synthesize, text, voice, 1.0, temperature, None)
        sf.write(out / f"{i:03d}.wav", result.samples, result.sample_rate)
        (out / f"{i:03d}.txt").write_text(text + "\n", encoding="utf-8")
        training_job["done"] += 1
        log.info("Trainingsmaterial %s: %d/%d", name, training_job["done"], training_job["total"])

    async def run() -> None:
        t0 = time.monotonic()
        try:
            await asyncio.gather(*(one(i, t) for i, t in enumerate(TRAINING_TEXTS, 1)))
            log.info("Trainingsmaterial %s fertig in %.0f s: %s", name, time.monotonic() - t0, out)
        except Exception as exc:
            training_job["error"] = str(exc)
            log.error("Trainingsmaterial %s fehlgeschlagen: %s", name, exc)
        finally:
            training_job["running"] = False

    asyncio.create_task(run())
    return training_job


@app.get("/api/training")
async def training_status():
    return training_job


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
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/overlay")
async def overlay():
    """Schlanke Seite für OBS-Browserquellen: nur Anzeige und Ton, keine
    Bedienelemente. Mit ?text=0 wird auch die Textanzeige ausgeblendet."""
    return FileResponse(STATIC_DIR / "overlay.html", headers=NO_CACHE)


app.mount("/static", FreshStaticFiles(directory=STATIC_DIR), name="static")
