"""FastAPI-Anwendung: REST-API, Audio-Stream, WebSocket-Status, statische GUI."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import io
import json
import logging
import os
import time
import re
import shutil
import sys
import tempfile
import wave
from collections import deque
from pathlib import Path

from fastapi import (FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import ConfigStore
from .player import PAUSE_BETWEEN_CHUNKS_S, Broadcaster, ChatSpeaker
from .stream import AudioStream
from .tts import TTS
from .twitch import TwitchChatClient
from .voices import (pool_voices, DEFAULT_VOICE, KOKORO_STOCK_REPO, VOICES, available_voices, clone_keys,
                     is_available, is_pocket, is_qwen, needs_download, ref_dir, sync_clone_voices,
                     display_name, preview_text, speed_instruct, voice_catalog)

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
for _engine, _count in (config.get("workers") or {}).items():
    if _engine in tts.POOL_ENGINES:
        tts.set_workers(_engine, _count, restart=False)
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
        "voices": available_voices(config.get("voice_labels"), config.get("voice_visibility")),
        "queue_size": speaker.queue_size,
        "test_active": speaker.test_active,
        "listeners": broadcaster.count,
        "stream_listeners": audio_stream.listener_count,
        "paused": audio_stream.paused,
        "last_perf": speaker.last_perf,
        "workers": tts.workers_state(),
    }


def _public_config() -> dict:
    """Konfiguration für die GUI – den Schlüssel der Produktiv-Instanz nie ausliefern."""
    data = config.as_dict()
    data["share_token_set"] = bool(data.pop("share_token", ""))
    return data


@app.get("/api/config")
async def get_config():
    return _public_config()


@app.put("/api/config")
async def put_config(changes: dict):
    try:
        config.update(changes)
        return _public_config()
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
    if ((tts.pool_active and is_qwen(voice)) or is_pocket(voice)) and not tts.warming:
        asyncio.create_task(asyncio.to_thread(tts.warmup, voice))
    return _status()


@app.post("/api/warmup")
async def warmup_voice(body: dict):
    """Qwen-Stimme schon beim Auswählen im Dropdown vorladen (vor "Wechseln").
    Läuft der Chat mit einem anderen Qwen-Modell, wird nicht umgeladen – das
    würde die laufende Ausgabe ausbremsen."""
    voice = str(body.get("voice", ""))
    if not is_available(voice) or tts.warming:
        return {"ok": False}
    if is_pocket(voice):  # eigene Prozesse, stört eine laufende Qwen-Ausgabe kaum
        asyncio.create_task(asyncio.to_thread(tts.warmup, voice))
        return {"ok": True}
    if not is_qwen(voice):
        return {"ok": False}
    current = config.get("voice")
    if (state["twitch"] is not None and is_available(current) and is_qwen(current)
            and VOICES[current].get("model") != VOICES[voice].get("model")):
        return {"ok": False, "reason": "chat"}
    asyncio.create_task(asyncio.to_thread(tts.warmup, voice))
    return {"ok": True}


@app.post("/api/preview")
async def preview():
    """Hörprobe: fester Satz mit der eingestellten Stimme, ohne Twitch-Chat."""
    voice = config.get("voice")
    return await say({"text": preview_text(voice if voice in VOICES else DEFAULT_VOICE)})


# ------------------------------------------------------- YouTube vorlesen
# Ein Video zur Zeit: herunterladen (720p), Text besorgen und übersetzen
# (app/youtube_prep.py als eigener Prozess), dann im Browser abspielen und
# die deutschen Sätze passend zur Position vorlesen. "Auswerfen" löscht alles.
YT_DIR = Path(os.environ.get("CONFIG_PATH", "/config/config.json")).parent / "youtube"
yt_job: dict = {"running": False, "log": deque(maxlen=12), "error": None, "proc": None, "url": ""}


def _yt_meta() -> dict | None:
    try:
        return json.loads((YT_DIR / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


async def _run_yt(url: str, original_level: float = 0.0, height: int = 720) -> None:
    env = {**os.environ, "HF_HUB_OFFLINE": "0", "PYTHONUNBUFFERED": "1"}
    last = ""
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "app.youtube_prep", "--url", url, "--out", str(YT_DIR),
            "--height", str(height),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env,
            start_new_session=True,  # eigene Prozessgruppe: Auswerfen beendet auch yt-dlp/ffmpeg
        )
        yt_job["proc"] = proc
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if not line:
                continue
            log.info("YouTube: %s", line)
            if line.startswith("SCHRITT: "):
                yt_job["log"].append(line[len("SCHRITT: "):])
            elif not line.startswith("ERGEBNIS "):
                last = line[-300:]
        code = await proc.wait()
        yt_job["proc"] = None
        meta = _yt_meta()
        if code != 0 or meta is None:
            yt_job["error"] = last or f"Abbruch mit Code {code}"
            return
        await _yt_dub(meta, original_level)
    except asyncio.CancelledError:
        yt_job["error"] = None
    except Exception as exc:
        log.exception("YouTube: Vertonen fehlgeschlagen")
        yt_job["error"] = str(exc)
    finally:
        yt_job["running"] = False
        yt_job["proc"] = None


def _yt_say(msg: str) -> None:
    yt_job["log"].append(msg)
    log.info("YouTube: %s", msg)


async def _yt_dub(meta: dict, original_level: float) -> None:
    """Alle Sätze mit der eingestellten Stimme vertonen (parallele Prozesse,
    hinter allen Chat-Nachrichten eingereiht), Tonspur bauen, ins Video setzen."""
    from . import klang, youtube_dub
    from .qwen_pool import with_priority

    cfg = config.as_dict()
    voice = speaker._voice(cfg)
    # Beim Vertonen bestimmt das Video das Tempo, nicht der Regler: jeder Satz
    # wird in Normaltempo gerechnet und nur so weit beschleunigt, wie er in
    # seine Lücke passen muss (youtube_dub.build_track) – ein Bearbeitungs-
    # schritt statt zwei (Regler + Anpassung) und keine unnötig gedehnten Sätze.
    _speed, temperature, instruct = speaker._gen_params({**cfg, "speed": 1.0})
    speed = 1.0
    clone_tempo = False
    sentences = meta["sentences"]
    n = len(sentences)
    # Stimmen-Wechsel bei erkannten Sprecherwechseln (">>" in den Untertiteln)
    pool = pool_voices(cfg) if cfg.get("voice_pool_youtube") else []
    speakers = len({s.get("sp", 0) for s in sentences})
    if pool and speakers > 1:
        _yt_say(f"{speakers} Sprecherwechsel-Abschnitte erkannt – wechsle zwischen {len(pool)} Stimmen")
    else:
        pool = []

    def voice_of(i: int) -> str:
        return pool[sentences[i].get("sp", 0) % len(pool)] if pool else voice
    _yt_say(f"Vertone {n} Sätze mit {VOICES.get(voice, {}).get('label', voice)} …")
    base = time.monotonic() + 7200  # Chat und Testtexte gehen immer vor
    sem = asyncio.Semaphore(max(2, tts.workers_for(voice) * 2))
    serial = asyncio.Lock()
    audios: list = [None] * n
    done = 0

    async def one(i: int) -> None:
        nonlocal done
        v = voice_of(i)
        # Engines ohne eigene Rechenprozesse (Kokoro) laufen im Hauptprozess:
        # dort nacheinander statt gleichzeitig
        lock = serial if not tts.parallel(v) else contextlib.nullcontext()
        async with sem:
            async with lock:
                r = await asyncio.to_thread(with_priority, (base, i), tts.synthesize,
                                            sentences[i]["text"], v, speed, temperature, instruct)
        samples = r.samples
        if clone_tempo:
            samples = await asyncio.to_thread(klang.apply_tempo, samples, r.sample_rate, speed,
                                              cfg.get("qwen_tempo", "auto"))
        if cfg.get("loudness_norm", True):
            samples = klang.normalize_loudness(samples, r.sample_rate)
        audios[i] = (samples, r.sample_rate)
        done += 1
        if done % max(1, n // 20) == 0 or done == n:
            yt_job["log"].append(f"Vertone … {done} von {n} Sätzen")

    t0 = time.monotonic()
    await asyncio.gather(*(one(i) for i in range(n)))
    _yt_say(f"Vertont in {(time.monotonic() - t0) / 60:.1f} min – baue deutsche Tonspur …")
    track, stats = await asyncio.to_thread(youtube_dub.build_track, sentences, audios, meta.get("duration"))
    dub = YT_DIR / "deutsch.wav"
    await asyncio.to_thread(youtube_dub.write_wav, dub, track)
    _yt_say("Setze Tonspur ins Video ein …")
    final = YT_DIR / "video_de.mp4"
    await asyncio.to_thread(youtube_dub.mux, YT_DIR / meta.get("video", "video.mp4"), dub, final, original_level)
    dub.unlink(missing_ok=True)
    (YT_DIR / meta.get("video", "video.mp4")).unlink(missing_ok=True)  # Platz sparen: Original steckt als 2. Spur drin
    meta.update(final="video_de.mp4", voice=voice, sped_up=stats["sped_up"],
                max_delay=round(stats["max_delay"], 1), original_level=original_level)
    (YT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    _yt_say(f"Fertig: {n} Sätze, {stats['sped_up']} beschleunigt, "
            f"höchstens {stats['max_delay']:.1f} s Versatz")


def _yt_wipe() -> None:
    shutil.rmtree(YT_DIR, ignore_errors=True)


@app.post("/api/youtube/prepare")
async def youtube_prepare(body: dict):
    url = str(body.get("url", "")).strip()
    if not url:
        raise HTTPException(400, "Bitte einen YouTube-Link eingeben.")
    if yt_job["running"]:
        raise HTTPException(409, "Es wird gerade schon ein Video vorbereitet.")
    await speaker.cancel_test()
    _yt_wipe()  # immer nur ein Video auf dem Server
    YT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        level = max(0.0, min(0.5, float(body.get("original_level", 0) or 0)))
    except (TypeError, ValueError):
        level = 0.0
    yt_job.update(running=True, error=None, url=url)
    yt_job["log"].clear()
    try:
        height = int(body.get("height", 720) or 720)
    except (TypeError, ValueError):
        height = 720
    height = height if height in (360, 480, 720, 1080, 1440, 2160) else 720
    yt_job["task"] = asyncio.create_task(_run_yt(url, level, height))
    return {"ok": True}


@app.get("/api/youtube/job")
async def youtube_job():
    meta = None if yt_job["running"] else _yt_meta()
    if meta is not None and not meta.get("final"):
        meta = None  # nur fertig vertonte Videos anbieten
    return {"running": yt_job["running"], "log": list(yt_job["log"]), "error": yt_job["error"],
            "meta": meta}


@app.get("/api/youtube/video")
async def youtube_video():
    meta = _yt_meta()
    path = YT_DIR / (meta or {}).get("final", "video_de.mp4")
    if meta is None or not meta.get("final") or not path.is_file():
        raise HTTPException(404, "Kein fertiges Video vorhanden.")
    # FileResponse beherrscht Range-Anfragen -> Spulen im Browser
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/youtube/export")
async def youtube_export():
    """Fertiges Video mit deutscher Tonspur herunterladen."""
    meta = _yt_meta()
    path = YT_DIR / (meta or {}).get("final", "video_de.mp4")
    if meta is None or not meta.get("final") or not path.is_file():
        raise HTTPException(404, "Kein fertiges Video vorhanden.")
    title = re.sub(r"[^\w\-. ]+", "", meta.get("title") or "video").strip()[:80] or "video"
    return FileResponse(path, media_type="video/mp4", filename=f"{title} (Deutsch KI).mp4")


@app.post("/api/youtube/eject")
async def youtube_eject():
    """Video samt Übersetzung verwerfen und vom Server löschen."""
    task = yt_job.get("task")
    if task is not None and not task.done():
        task.cancel()
    proc = yt_job.get("proc")
    if proc is not None:
        import signal

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        await asyncio.sleep(0.3)
    await speaker.cancel_test()
    _yt_wipe()
    yt_job.update(error=None)
    yt_job["log"].clear()
    return {"ok": True}


@app.post("/api/youtube/say")
async def youtube_say(body: dict):
    text = str(body.get("text", "")).strip()
    if not text:
        return {"ok": False}
    if state["twitch"] is not None:
        raise HTTPException(409, "YouTube vorlesen ist gesperrt, solange der Chat vorgelesen wird.")
    if not is_available(config.get("voice")):
        raise HTTPException(400, "Die eingestellte Stimme ist nicht verfügbar.")
    await speaker.say_youtube(text)
    return {"ok": True}


@app.post("/api/youtube/prefetch")
async def youtube_prefetch(body: dict):
    texts = [str(t).strip() for t in (body.get("texts") or []) if str(t).strip()][:8]
    if state["twitch"] is not None or not texts:
        return {"started": 0}
    return {"started": speaker.prefetch_texts(texts)}


@app.post("/api/youtube/clear")
async def youtube_clear():
    """Gespult/gestoppt: wartende und laufende Sätze verwerfen."""
    await speaker.cancel_test()
    return {"ok": True}


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

    # "direct"/0 = sofort, 1–6 = so viele Abschnitte vorab, "all"/-1 = komplett
    raw_pb = body.get("prebuffer")
    if raw_pb in (None, ""):
        prebuffer = None
    elif str(raw_pb).lower() in ("all", "kplt", "-1"):
        prebuffer = -1
    elif str(raw_pb).lower() == "direct":
        prebuffer = 0
    else:
        try:
            prebuffer = max(0, min(6, int(raw_pb)))
        except (TypeError, ValueError):
            raise HTTPException(400, "Ungültiger Wert für den Vorlauf.")
    await speaker.say_text(text, prebuffer)
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
    # Jeder Abschnitt wartet in einem eigenen Thread auf einen freien
    # Rechenprozess. Der Standard-Threadpool (32) war bei langen Texten voll –
    # andere Aufrufe (Stimmenliste, Laden) standen dann dahinter an.
    from concurrent.futures import ThreadPoolExecutor

    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=int(os.environ.get("APP_THREADS", "128") or 128)))
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
    # Korrektur auch beim gewählten Ausschnitt merken (falls später neu gewählt)
    chosen = _cand_dir(name) / "gewaehlt"
    if chosen.exists():
        idx = chosen.read_text().strip() or "1"
        if (_cand_dir(name) / f"{idx}.txt").exists():
            (_cand_dir(name) / f"{idx}.txt").write_text(text + "\n", encoding="utf-8")
    if clone_job.get("result") and clone_job["result"].get("name") == name:
        clone_job["result"]["text"] = text
    return {"ok": True}


def _cand_dir(name: str) -> Path:
    return ref_dir() / ".kandidaten" / name


@app.get("/api/clone/candidates/{name}")
async def clone_candidates(name: str):
    """Ausschnitte, die beim Erstellen zur Auswahl gespeichert wurden."""
    name = name.strip().lower()
    folder = _cand_dir(name)
    if not CLONE_NAME.match(name) or not folder.is_dir():
        return {"name": name, "chosen": None, "candidates": []}
    chosen_file = folder / "gewaehlt"
    chosen = int(chosen_file.read_text().strip() or 1) if chosen_file.exists() else None
    out = []
    for wav in sorted(folder.glob("*.wav"), key=lambda p: int(p.stem) if p.stem.isdigit() else 99):
        if not wav.stem.isdigit():
            continue
        txt = wav.with_suffix(".txt")
        seconds = max(0.0, (wav.stat().st_size - 44) / (24000 * 2))  # 24 kHz, 16 Bit mono
        out.append({"idx": int(wav.stem), "seconds": round(seconds, 1),
                    "text": txt.read_text(encoding="utf-8").strip() if txt.exists() else ""})
    return {"name": name, "chosen": chosen, "candidates": out}


@app.get("/api/clone/candidates/{name}/{idx}.wav")
async def clone_candidate_audio(name: str, idx: int):
    name = name.strip().lower()
    wav = _cand_dir(name) / f"{idx}.wav"
    if not CLONE_NAME.match(name) or not wav.is_file():
        raise HTTPException(404, "Ausschnitt nicht gefunden.")
    return FileResponse(wav, media_type="audio/wav", headers=NO_CACHE)


@app.post("/api/clone/candidates/{name}/{idx}")
async def clone_choose(name: str, idx: int):
    """Anderen Ausschnitt als Vorlage der eigenen Stimme übernehmen. Alle
    Engines merken das an der geänderten Datei und bereiten neu vor."""
    name = name.strip().lower()
    folder = _cand_dir(name)
    wav, txt = folder / f"{idx}.wav", folder / f"{idx}.txt"
    if not CLONE_NAME.match(name) or not wav.is_file():
        raise HTTPException(404, "Ausschnitt nicht gefunden.")
    if clone_job.get("running") and clone_job.get("name") == name:
        raise HTTPException(409, "Diese Stimme wird gerade erstellt.")
    tmp = ref_dir() / f".{name}.wav.tmp"
    shutil.copyfile(wav, tmp)
    if txt.exists():
        shutil.copyfile(txt, ref_dir() / f"{name}.txt")
    os.replace(tmp, ref_dir() / f"{name}.wav")
    (folder / "gewaehlt").write_text(f"{idx}\n", encoding="utf-8")
    tts.forget_clone(name)
    sync_clone_voices()
    if clone_job.get("result") and clone_job["result"].get("name") == name:
        clone_job["result"]["chosen"] = idx
    log.info("Eigene Stimme %s: Ausschnitt %d übernommen", name, idx)
    await broadcaster.send({"type": "voices_changed"})
    text = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
    return {"ok": True, "chosen": idx, "text": text}


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
    vis = dict(config.get("voice_visibility") or {})
    if any(k in vis for k in keys):
        changes["voice_visibility"] = {k: v for k, v in vis.items() if k not in keys}
    if changes:
        config.update(changes)
    for suffix in (".wav", ".txt"):
        (folder / f"{name}{suffix}").unlink(missing_ok=True)
    (folder / "kokoro" / f"{name}.pt").unlink(missing_ok=True)
    shutil.rmtree(_cand_dir(name), ignore_errors=True)
    sync_clone_voices()
    tts.forget_clone(name)
    log.info("Eigene Stimme %s gelöscht", name)
    await broadcaster.send({"type": "voices_changed"})
    return {"ok": True, "voice": config.get("voice")}


# ------------------------------------------------- Rechenprozesse
@app.post("/api/workers")
async def set_workers(body: dict):
    """Anzahl der Rechenprozesse einer Engine setzen und Prozesse neu starten."""
    engine = str(body.get("engine", ""))
    if engine not in tts.POOL_ENGINES:
        raise HTTPException(400, "Unbekannte Engine.")
    try:
        count = min(max(int(body.get("count", 1)), 1), 6)
    except (TypeError, ValueError):
        raise HTTPException(400, "Ungültige Anzahl.")
    config.update({"workers": {engine: count}})
    speaker.skip()  # laufende Nachricht hätte sonst einen Fehler
    await asyncio.to_thread(tts.set_workers, engine, count)
    voice = config.get("voice")
    # Qwen-Prozesse gleich wieder hochfahren (Modell laden), sonst wartet die
    # erste Nachricht darauf
    if (is_available(voice) and not tts.warming
            and ((engine == "qwen" and is_qwen(voice)) or (engine == "pocket" and is_pocket(voice)))):
        asyncio.create_task(asyncio.to_thread(tts.warmup, voice))
    log.info("Rechenprozesse %s: %d", engine, count)
    return {"ok": True, "workers": tts.workers_state()}


# ------------------------------------------------- Stimmen verwalten
downloads: dict[str, str] = {}  # Stimmen-Schlüssel -> "läuft" / Fehlertext


@app.get("/api/voices/catalog")
async def voices_catalog():
    groups = await asyncio.to_thread(voice_catalog, config.get("voice_labels"),
                                     config.get("voice_visibility"),
                                     config.get("shared_voices"))
    for g in groups:
        for v in g["voices"]:
            if v["key"] in downloads:
                v["download_state"] = downloads[v["key"]]
    return groups


async def _download_voicepack(key: str) -> None:
    """Kokoro-Original-Stimmpaket einmalig in den HF-Cache laden (eigener
    Prozess mit Internetzugang, der Container läuft sonst offline)."""
    pack = VOICES[key]["voicepack"]
    downloads[key] = "läuft"
    code = ("import sys; from huggingface_hub import hf_hub_download; "
            "print(hf_hub_download(sys.argv[1], f'voices/{sys.argv[2]}.pt'))")
    env = {**os.environ, "HF_HUB_OFFLINE": "0"}
    env.pop("HF_TOKEN", None)  # öffentliches Repo
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code, KOKORO_STOCK_REPO, pack,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
        out, err = await asyncio.wait_for(proc.communicate(), timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(err.decode(errors="replace").strip()[-300:])
        downloads.pop(key, None)
        log.info("Stimmpaket %s geladen: %s", pack, out.decode().strip().splitlines()[-1])
    except Exception as exc:
        downloads[key] = f"Download fehlgeschlagen: {exc}"
        log.warning("Stimmpaket %s nicht ladbar: %s", pack, exc)
    await broadcaster.send({"type": "voices_changed"})


@app.put("/api/voices/visibility")
async def voices_visibility(body: dict):
    """Stimmen ein-/ausblenden ({Schlüssel: true/false}). Aktivierte
    Kokoro-Katalogstimmen werden dabei nachgeladen. Wird die gewählte Stimme
    ausgeblendet, springt die Auswahl auf die erste sichtbare Stimme."""
    sync_clone_voices()
    vis = dict(config.get("voice_visibility") or {})
    for key, on in body.items():
        if key in VOICES:
            vis[key] = bool(on)
            if on and needs_download(key) and downloads.get(key) != "läuft":
                asyncio.create_task(_download_voicepack(key))
    changes: dict = {"voice_visibility": vis}
    if not vis.get(config.get("voice"), True):
        usable = available_voices(None, vis)
        if usable:
            changes["voice"] = DEFAULT_VOICE if any(v["key"] == DEFAULT_VOICE for v in usable) \
                else usable[0]["key"]
            speaker.skip()
    config.update(changes)
    await broadcaster.send({"type": "voices_changed"})
    return {"ok": True, "voice": config.get("voice")}


# ------------------------------------- Freigabe für die Produktiv-Instanz
# Die Produktiv-Instanz ("Freunde") bekommt nur die ausdrücklich freigegebenen
# und übernommenen Stimmen (zweites Häkchen in "Stimmen verwalten", dann "Für
# Produktiv übernehmen"). Alles andere – Klonen, Training, nicht freigegebene
# eigene Stimmen – bleibt unerreichbar.
share_lock = asyncio.Lock()  # eine Anfrage nach der anderen, schont die CPU


def _check_share_token(request: Request) -> None:
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    expected = config.get("share_token") or ""
    if not token or not expected or not hmac.compare_digest(token, expected):
        raise HTTPException(401, "Ungültiger Schlüssel.")


def _voice_list(keys) -> list[dict]:
    sync_clone_voices()
    labels = config.get("voice_labels")
    out = []
    for key in keys or []:
        if key in VOICES and is_available(key):
            out.append({"key": key, "label": display_name(key, labels),
                        "lang": VOICES[key].get("lang", "d")})
    return out


def _share_state() -> dict:
    draft = sorted(config.get("shared_voices") or [])
    published = sorted(config.get("shared_published") or [])
    return {"draft": len(draft), "published": len(published), "pending": draft != published,
            "configured": bool(config.get("prod_url") and config.get("share_token"))}


@app.put("/api/voices/shared")
async def voices_shared(body: dict):
    """Häkchen "Freigeben" setzen ({Schlüssel: true/false}) – wirkt erst
    nach "Für Produktiv übernehmen"."""
    sync_clone_voices()
    shared = set(config.get("shared_voices") or [])
    for key, on in body.items():
        if key in VOICES and on:
            shared.add(key)
        else:
            shared.discard(key)
    config.update({"shared_voices": sorted(shared)})
    return {"ok": True, **_share_state()}


@app.get("/api/share/state")
async def share_state():
    return _share_state()


@app.post("/api/share/publish")
async def share_publish(body: dict):
    """Freigegebene Stimmen an die Produktiv-Instanz übertragen. body.self_url:
    Adresse dieser Instanz aus Sicht der Produktiv-Instanz."""
    import httpx

    self_url = str(body.get("self_url") or "").strip().rstrip("/")
    if self_url:
        config.update({"share_self_url": self_url})
    prod, token, self_url = config.get("prod_url"), config.get("share_token"), config.get("share_self_url")
    if not (prod and token and self_url):
        raise HTTPException(400, "Bitte Adresse der Produktiv-Instanz, Schlüssel und eigene Adresse eintragen.")
    keys = sorted(config.get("shared_voices") or [])
    voices = await asyncio.to_thread(_voice_list, keys)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"{prod}/api/remote/sync",
                                  headers={"Authorization": f"Bearer {token}"},
                                  json={"url": self_url, "voices": voices})
    except Exception as exc:
        raise HTTPException(502, f"Produktiv-Instanz nicht erreichbar: {exc}")
    if r.status_code == 401:
        raise HTTPException(502, "Produktiv-Instanz lehnt den Schlüssel ab.")
    if r.status_code == 404:
        raise HTTPException(502, "Produktiv-Instanz kennt die Freigabe noch nicht – bitte dort aktualisieren.")
    if r.status_code != 200:
        raise HTTPException(502, f"Produktiv-Instanz: Fehler {r.status_code}")
    config.update({"shared_published": [v["key"] for v in voices]})
    log.info("Freigabe übernommen: %d Stimmen an %s", len(voices), prod)
    return {"ok": True, **_share_state()}


@app.get("/api/share/voices")
async def share_voices(request: Request):
    _check_share_token(request)
    return {"voices": await asyncio.to_thread(_voice_list, config.get("shared_published"))}


@app.post("/api/share/synthesize")
async def share_synthesize(request: Request, body: dict):
    """Text mit einer freigegebenen Stimme erzeugen, Antwort: WAV (mono, 16 bit)."""
    _check_share_token(request)
    voice = str(body.get("voice") or "")
    text = str(body.get("text") or "").strip()[:1000]
    speed = min(max(float(body.get("speed") or 1.0), 0.5), 2.0)
    if not text:
        raise HTTPException(400, "Kein Text.")
    if voice not in (config.get("shared_published") or []):
        raise HTTPException(403, "Diese Stimme ist nicht freigegeben.")
    sync_clone_voices()
    if voice not in VOICES or not is_available(voice):
        raise HTTPException(404, "Stimme hier nicht verfügbar.")

    import numpy as np

    # Neutraler Stil für die Freunde – Sprechtempo bei Qwen-Presets per Anweisung
    temperature = config.get("qwen_temperature")
    instruct = speed_instruct(None, speed)
    chunks = ChatSpeaker._chunks(text, voice)
    t0 = time.perf_counter()
    async with share_lock:
        try:
            jobs = [asyncio.to_thread(tts.synthesize, c, voice, speed, temperature, instruct)
                    for c in chunks]
            if tts.parallel(voice):
                results = await asyncio.gather(*jobs)
            else:
                results = [await j for j in jobs]
        except Exception as exc:
            log.warning("Freigabe: %s fehlgeschlagen: %s", voice, exc)
            raise HTTPException(500, f"Erzeugung fehlgeschlagen: {exc}")
    sr = results[0].sample_rate
    gap = np.zeros(int(sr * PAUSE_BETWEEN_CHUNKS_S), dtype=np.float32)
    parts = []
    for i, r in enumerate(results):
        if i:
            parts.append(gap)
        parts.append(np.asarray(r.samples, dtype=np.float32))
    pcm = (np.clip(np.concatenate(parts), -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    log.info("Freigabe: %s, %d Zeichen in %.1f s", voice, len(text), time.perf_counter() - t0)
    return Response(buf.getvalue(), media_type="audio/wav")


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


@app.post("/api/pause")
async def pause(body: dict | None = None):
    """Vorlesen anhalten/fortsetzen (Chat und Testtext). Ohne "paused" im
    Körper wird umgeschaltet. Neue Nachrichten werden weiter angenommen und
    vorab gerechnet; die Warteschlangen-Grenze gilt wie sonst."""
    want = (body or {}).get("paused")
    audio_stream.paused = (not audio_stream.paused) if want is None else bool(want)
    await broadcaster.send({"type": "paused", "paused": audio_stream.paused})
    return {"paused": audio_stream.paused}


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
