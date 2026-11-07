"""Message queue + sequential playback orchestration.

Flow: Twitch message -> filters (bot list, per-user cooldown, cleaning)
-> bounded queue (drop oldest on overflow) -> worker synthesizes via the
TTS engine -> samples go into the continuous audio stream (stream.py, played
by the browser via <audio src="/stream">) -> worker waits until the stream has
emitted the whole utterance (or skip) before taking the next message.
Guarantees strictly sequential playback, no overlap. The WebSocket only
carries status/text for the UI, no audio.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

from .config import ConfigStore
from . import thirdparty_emotes
from .processing import clean_message, has_mention, is_speakable
from .stream import AudioStream
from .tts import QWEN_FIRST_CHUNK_CHARS, TTS, split_for_qwen
from .voices import QWEN_STYLES, VOICES, speed_instruct, available_voices, chunk_chars, is_available, is_chunked, is_pocket, is_qwen

# Pause zwischen zwei Abschnitten eines langen Qwen-Texts
PAUSE_BETWEEN_CHUNKS_S = 0.25
# Mehrere Rechenprozesse: Texte bis zu dieser Länge (Zeichen, inkl. Name)
# gelten als kurz und werden bevorzugt gerechnet.
SHORT_CHARS = 60
# Namensansage als eigener Abschnitt erst ab dieser Textlänge (Zeichen)
NAME_SPLIT_MIN_CHARS = 25

log = logging.getLogger(__name__)


@dataclass
class QueuedMessage:
    username: str
    text: str
    raw: bool = False  # Testtext aus der GUI: ohne Namensansage vorlesen
    arrived: float = field(default_factory=time.monotonic)  # für die Wartezeit-Anzeige
    batched: bool = False  # steckt in einer (laufenden/fertigen) Gruppenberechnung


def _trim_silence(samples, sr: int, head: bool, tail: bool, keep_s: float = 0.05):
    """Stille am Anfang/Ende abschneiden (Pegel unter 3 % des Maximums),
    mit kurzem Rest, damit nichts hart abreißt."""
    import numpy as np

    if len(samples) == 0:
        return samples
    win = max(1, int(sr * 0.01))
    n = len(samples) // win
    if n < 3:
        return samples
    rms = np.sqrt(np.mean(samples[: n * win].reshape(n, win) ** 2, axis=1))
    loud = np.nonzero(rms > 0.03 * (rms.max() or 1.0))[0]
    if len(loud) == 0:
        return samples
    keep = int(sr * keep_s)
    start = max(0, loud[0] * win - keep) if head else 0
    end = min(len(samples), (loud[-1] + 1) * win + keep) if tail else len(samples)
    return samples[start:end]


class Broadcaster:
    """Tracks connected WebSocket UIs and broadcasts JSON messages."""

    def __init__(self) -> None:
        self._clients: set = set()

    def add(self, ws) -> None:
        self._clients.add(ws)

    def remove(self, ws) -> None:
        self._clients.discard(ws)

    @property
    def count(self) -> int:
        return len(self._clients)

    async def send(self, payload: dict) -> None:
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.remove(ws)


class ChatSpeaker:
    def __init__(
        self, config: ConfigStore, tts: TTS, broadcaster: Broadcaster, audio: AudioStream
    ) -> None:
        self.config = config
        self.tts = tts
        self.broadcaster = broadcaster
        self.audio = audio
        self._queue: deque[QueuedMessage] = deque()
        self._queue_event = asyncio.Event()
        self._last_spoken: dict[str, float] = {}
        self._worker: asyncio.Task | None = None
        self._skip = asyncio.Event()
        self._current_id = 0
        self._current_raw = False  # läuft gerade ein Testtext?
        # Leistungswerte der letzten Ausgabe (Twitch oder Testtext) für die GUI
        self.last_perf: dict | None = None
        self.dropped_total = 0
        self._emote_fetches: set[str] = set()
        self._prefetch_tasks: set[asyncio.Future] = set()
        self._announced: QueuedMessage | None = None  # "Erzeuge Sprache" schon gezeigt

    def _ensure_emote_fetch(self, room_id: str) -> None:
        """Emote-Listen einmal pro Kanal im Hintergrund laden."""
        if room_id in self._emote_fetches:
            return
        self._emote_fetches.add(room_id)

        async def _run() -> None:
            try:
                await thirdparty_emotes.fetch_names(room_id)
            except Exception as exc:
                log.warning("Emote-Listen konnten nicht geladen werden: %s", exc)
                self._emote_fetches.discard(room_id)  # nächster Versuch erlaubt

        asyncio.create_task(_run())

    # ---------------------------------------------------------------- intake
    async def on_message(
        self, username: str, text: str, emotes_tag: str | None, room_id: str | None = None
    ) -> None:
        cfg = self.config.as_dict()
        user_key = username.lower()

        if user_key in cfg["bot_blocklist"]:
            return

        # Erwähnungen abgeschaltet: ganze Nachricht überspringen.
        # Vor dem Cooldown-Stempel, damit eine übersprungene Nachricht die
        # Wartezeit des Users nicht zurücksetzt.
        if not cfg["read_mentions"] and has_mention(text):
            return

        now = time.monotonic()
        last = self._last_spoken.get(user_key, 0.0)
        if now - last < cfg["cooldown_seconds"]:
            return

        spoken = clean_message(text, emotes_tag, cfg["read_emotes"], cfg["read_smileys"])

        # Emotes von BTTV/7TV/FFZ stehen nicht in den Twitch-Tags und müssen
        # anhand der Namenslisten entfernt werden. Der Abruf läuft einmal pro
        # Kanal im Hintergrund; bis er fertig ist, greift der Filter noch nicht.
        if not cfg["read_emotes"] and room_id and spoken:
            names = thirdparty_emotes.cached_names(room_id)
            if names is None:
                self._ensure_emote_fetch(room_id)
            else:
                spoken = thirdparty_emotes.strip_names(spoken, names)
        # Nichts Sprechbares übrig (nur Satz-/Sonderzeichen oder ein nicht
        # erkanntes Emoji)? Dann gar nicht erst vorlesen – sonst bliebe nur
        # der Name übrig.
        if not is_speakable(spoken):
            return

        self._last_spoken[user_key] = now
        self._queue.append(QueuedMessage(username=username, text=spoken))
        # Queue limit: drop oldest instead of building up delay.
        # Nachrichten, die schon in einer Gruppe berechnet werden, bleiben
        # geschützt – sonst wäre die Rechenzeit verloren. Verworfen wird die
        # älteste noch nicht berechnete.
        limit = cfg["queue_limit"]
        while len(self._queue) > limit:
            victim = next((m for m in self._queue if not m.batched), None)
            if victim is None:
                break
            self._queue.remove(victim)
            self.dropped_total += 1
            log.info("Warteschlange voll (%d): verworfen %s: »%s«", limit,
                     victim.username, victim.text[:60])
        self._queue_event.set()
        self._maybe_prefetch()
        await self._push_queue_state()

    async def say_text(self, text: str) -> None:
        """Testtext aus der GUI vorlesen – ohne Twitch, ohne Filter, Cooldown
        und Queue-Limit, ohne Namensansage. Reiht sich normal in die
        Warteschlange ein, läuft also nie über eine Chat-Nachricht drüber."""
        self.start()
        self._queue.append(QueuedMessage(username="Test", text=text, raw=True))
        self._queue_event.set()
        self._maybe_prefetch()
        await self._push_queue_state()

    # ---------------------------------------------------------------- worker
    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run(), name="tts-worker")

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        self._queue.clear()
        self._last_spoken.clear()
        self._skip.set()  # release a potentially waiting playback
        self.audio.stop_current()
        await self.broadcaster.send({"type": "stop_audio"})
        await self._push_queue_state()

    def skip(self) -> None:
        """Abort the currently playing message, jump to the next one."""
        self._skip.set()

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    async def _push_queue_state(self) -> None:
        await self.broadcaster.send({"type": "queue", "size": len(self._queue)})

    async def _run(self) -> None:
        while True:
            while not self._queue:
                self._queue_event.clear()
                await self._queue_event.wait()
            self._maybe_prefetch()
            msg = self._pick_next()
            if msg is None:
                # Alles Wartende rechnet noch -> auf die nächste fertige warten.
                # Die Oberfläche zeigt schon jetzt "Erzeuge Sprache …" für die
                # älteste wartende Nachricht (sonst wirkt der Klick tot).
                head = self._queue[0] if self._queue else None
                if head is not None and self._announced is not head:
                    self._announced = head
                    await self.broadcaster.send(
                        {"type": "generating", "username": head.username, "text": head.text})
                if self._prefetch_tasks:
                    await asyncio.wait(set(self._prefetch_tasks),
                                       return_when=asyncio.FIRST_COMPLETED)
                continue
            self._queue.remove(msg)
            await self._push_queue_state()
            try:
                await self._speak(msg)
            except Exception as exc:
                log.error("TTS für Nachricht von %s fehlgeschlagen: %s", msg.username, exc)
                await self.broadcaster.send({"type": "error", "message": str(exc)})

    def _pick_next(self) -> QueuedMessage | None:
        """Welche Nachricht als Nächstes? Mit einem Rechenprozess strikt der
        Reihe nach. Mit mehreren: die älteste, deren Audio schon fertig ist –
        kurze Nachrichten überholen so lange, die noch rechnen. Gespielt wird
        trotzdem immer nur eine Nachricht nach der anderen."""
        cfg = self.config.as_dict()
        try:
            voice = self._voice(cfg)
        except RuntimeError:
            return self._queue[0]
        if not self.tts.parallel(voice):
            return self._queue[0]
        _, temperature, instruct = self._gen_params(cfg)
        # Nur Nachrichten, deren Anfang komplett fertig ist – bei abgetrennter
        # Namensansage Name UND erster Textabschnitt. "X sagt:" startet also
        # nie vor dem Text, sonst entsteht eine unnatürliche Lücke.
        for m in self._queue:
            chunks = self._message_chunks(m, cfg, voice)
            head = chunks[:2] if self._is_name_chunk(chunks[0], m) and len(chunks) > 1 else chunks[:1]
            if all(self.tts.has_result(t, voice, temperature, instruct) for t in head):
                return m
        if self._prefetch_tasks:
            return None  # gleich ist etwas fertig
        # Nichts fertig und nichts in Arbeit (z. B. Fehler): älteste direkt rechnen
        return self._queue[0]

    async def cancel_test(self) -> None:
        """Testtexte abbrechen: wartende aus der Queue nehmen, laufenden stoppen.
        Twitch-Nachrichten bleiben unberührt."""
        self._queue = deque(m for m in self._queue if not m.raw)
        if self._current_raw:
            self._skip.set()
        else:  # nur wartende entfernt -> Oberfläche (Vorlesen-Knopf) freigeben
            await self.broadcaster.send({"type": "stop_audio"})
        await self._push_queue_state()

    # ------------------------------------------------- Gruppenberechnung
    def _voice(self, cfg) -> str:
        voice = cfg["voice"]
        if not is_available(voice):
            usable = available_voices()
            if not usable:
                raise RuntimeError(
                    "Keine Stimmen installiert – bitte install.sh erneut ausführen."
                )
            voice = usable[0]["key"]
        return voice

    @staticmethod
    def _spoken_text(msg: QueuedMessage, cfg, voice: str) -> str:
        if cfg["read_username"] and not msg.raw:
            if cfg["username_style"] == "sagt":
                # Verb passend zur Sprache der gewählten Stimme
                verb = "sagt" if VOICES[voice]["lang"] == "d" else "says"
                return f"{msg.username} {verb} {msg.text}"
            return f"{msg.username}: {msg.text}"
        return msg.text

    @staticmethod
    def _gen_params(cfg) -> tuple:
        style = cfg["qwen_style"]
        instruct = cfg["qwen_style_custom"] if style == "eigen" else QWEN_STYLES.get(style)
        # Qwen-Tempo nicht nachträglich strecken (klingt verfremdet), sondern
        # bei den Preset-Stimmen per Anweisung; eigene Stimmen sprechen im
        # Tempo ihrer Referenzaufnahme.
        return cfg["speed"], cfg["qwen_temperature"], speed_instruct(instruct, cfg["speed"])

    @staticmethod
    def _chunks(text: str, voice: str) -> list[str]:
        limit = chunk_chars(voice)
        if not is_chunked(voice):
            chunks = [text]
        elif limit:
            chunks = split_for_qwen(text, max_chars=limit,
                                    first_max=min(QWEN_FIRST_CHUNK_CHARS, limit))
        else:
            chunks = split_for_qwen(text, first_max=QWEN_FIRST_CHUNK_CHARS)
        return chunks or [text]

    @staticmethod
    def _is_name_chunk(chunk: str, msg: QueuedMessage) -> bool:
        return not msg.raw and chunk.startswith(msg.username) and chunk.endswith(":") \
            and len(chunk) <= len(msg.username) + 6

    def _message_chunks(self, msg: QueuedMessage, cfg, voice: str) -> list[str]:
        """Abschnitte einer Nachricht. Mit mehreren Rechenprozessen wird die
        Namensansage ("X sagt:") ein eigener, kurzer Abschnitt: Er ist schnell
        fertig (und bei Stammgästen schon im Zwischenspeicher), der eigentliche
        Text rechnet parallel – der erste Ton kommt so deutlich früher.
        Sehr kurze Nachrichten bleiben am Stück (einzelne Wörter verschluckt
        Qwen sonst eher)."""
        if (cfg["read_username"] and not msg.raw and self.tts.parallel(voice)
                and len(msg.text) >= NAME_SPLIT_MIN_CHARS):
            if cfg["username_style"] == "sagt":
                verb = "sagt" if VOICES[voice]["lang"] == "d" else "says"
                prefix = f"{msg.username} {verb}:"
            else:
                prefix = f"{msg.username}:"
            return [prefix] + self._chunks(msg.text, voice)
        return self._chunks(self._spoken_text(msg, cfg, voice), voice)

    def _maybe_prefetch(self) -> None:
        """Qwen mit mehreren Rechenprozessen (CPU): wartende Nachrichten
        sofort an freie Prozesse verteilen – auch während noch eine Nachricht
        spielt. So hält der CT mit einem vollen Chat besser Schritt."""
        # Darf nie nach außen werfen: läuft u. a. direkt im Twitch-Empfang –
        # ein Fehler hier hat sonst die Chat-Verbindung gekappt.
        try:
            self._prefetch_inner()
        except Exception as exc:
            log.warning("Gruppenberechnung nicht gestartet: %s", exc)

    def _prefetch_inner(self) -> None:
        if not self._queue:
            return
        cfg = self.config.as_dict()
        try:
            voice = self._voice(cfg)
        except RuntimeError:
            return
        if not self.tts.parallel(voice):
            return
        free = self.tts.workers_for(voice) - len(self._prefetch_tasks)
        if free <= 0:
            return
        # Kurze Nachrichten zuerst (schnell hörbar), sonst der Reihe nach.
        # Gespielt wird später die älteste fertige.
        candidates = [(m, self._message_chunks(m, cfg, voice))
                      for m in self._queue if not m.batched]
        # stabil sortiert: innerhalb von kurz/lang bleibt die Reihenfolge
        candidates.sort(key=lambda c: len(c[0].text) > SHORT_CHARS)
        _, temperature, instruct = self._gen_params(cfg)
        for m, chunks in candidates:
            if free <= 0:
                break
            m.batched = True
            # alle Abschnitte der Nachricht gleichzeitig anstoßen
            for text in chunks:
                task = asyncio.ensure_future(asyncio.to_thread(
                    self.tts.prefetch_one, text, voice, temperature, instruct))
                self._prefetch_tasks.add(task)
                task.add_done_callback(self._prefetch_done)
            free -= len(chunks)

    def _prefetch_done(self, task) -> None:
        self._prefetch_tasks.discard(task)
        self._maybe_prefetch()

    async def _speak(self, msg: QueuedMessage) -> None:
        self._current_raw = msg.raw
        try:
            await self._speak_inner(msg)
        finally:
            self._current_raw = False

    async def _speak_inner(self, msg: QueuedMessage) -> None:
        cfg = self.config.as_dict()
        voice = self._voice(cfg)
        # Qwen-Texte abschnittsweise: Der erste Abschnitt wird schon gespielt,
        # während die nächsten im Hintergrund rechnen – lange Texte starten so
        # nach Sekunden statt erst nach der kompletten Berechnung. Kokoro ist
        # schnell genug und bekommt den Text am Stück.
        chunks = self._message_chunks(msg, cfg, voice)
        speed, temperature, instruct = self._gen_params(cfg)

        self._skip.clear()  # ältere Klicks zählen nicht für diese Nachricht
        # Anzeige "Erzeuge Sprache …" – bei langsamen Engines (Qwen auf CPU)
        # sonst sekundenlang kein Lebenszeichen in der GUI. Wurde sie für diese
        # Nachricht schon beim Warten gesendet, nicht neu starten (Zähler).
        if self._announced is not msg:
            await self.broadcaster.send(
                {"type": "generating", "username": msg.username, "text": msg.text}
            )
        self._announced = msg

        def synth(chunk: str):
            return asyncio.ensure_future(
                asyncio.to_thread(self.tts.synthesize, chunk, voice, speed, temperature, instruct)
            )

        totals = {"compute": 0.0, "load": 0.0, "audio": 0.0, "decode": 0.0,
                  "batch": 1, "wait": None}
        all_cached = True
        t_start = time.perf_counter()
        first_audio_s = None
        utterance_id = None
        # Mehrere Rechenprozesse: alle Abschnitte gleichzeitig anstoßen
        # (gespielt wird trotzdem der Reihe nach). Sonst immer nur den
        # nächsten, während der aktuelle spielt.
        parallel = self.tts.parallel(voice)
        pendings = [synth(c) for c in chunks] if parallel else [synth(chunks[0])]
        skipped = False

        for idx in range(len(chunks)):
            result, skipped = await self._wait_or_skip(pendings[idx])
            if skipped:
                break
            if not parallel and idx + 1 < len(chunks):
                pendings.append(synth(chunks[idx + 1]))

            totals["compute"] += result.compute_seconds
            totals["load"] += result.load_seconds
            totals["audio"] += result.duration_seconds
            totals["decode"] += result.decode_seconds
            totals["batch"] = max(totals["batch"], getattr(result, "batch", 1))
            all_cached &= result.cached
            self.last_perf = self._perf(cfg, voice, totals, all_cached, first_audio_s)

            if idx == 0:
                first_audio_s = time.perf_counter() - t_start
                totals["wait"] = time.monotonic() - msg.arrived
                if not msg.raw:
                    log.info("Wartezeit: %.1f s nach Eingang (%s, %d wartend)",
                             totals["wait"], msg.username, len(self._queue))
                self.last_perf = self._perf(cfg, voice, totals, all_cached, first_audio_s)
                self._current_id += 1
                utterance_id = self._current_id
                # Nur Anzeige für die UI/das Overlay – der Ton läuft über /stream.
                await self.broadcaster.send(
                    {
                        "type": "speak",
                        "id": utterance_id,
                        "username": msg.username,
                        "text": msg.text,
                        "perf": self.last_perf,
                    }
                )

            samples = result.samples
            name_split = self._is_name_chunk(chunks[0], msg)
            if name_split and idx == 0:
                # Namensansage: Stille am Ende weg, der Text schließt direkt an
                samples = _trim_silence(samples, result.sample_rate, head=False, tail=True)
            elif name_split and idx == 1:
                samples = _trim_silence(samples, result.sample_rate, head=True, tail=False)
            elif idx > 0:  # kurze Atempause zwischen den Abschnitten
                import numpy as np

                pause = np.zeros(int(PAUSE_BETWEEN_CHUNKS_S * result.sample_rate), dtype=np.float32)
                samples = np.concatenate([pause, samples])
            _, skipped = await self._wait_or_skip(self.audio.play(samples))
            if skipped:
                break

        if skipped:
            self.audio.stop_current()
            await self.broadcaster.send({"type": "stop_audio"})
            return
        self.last_perf = self._perf(cfg, voice, totals, all_cached, first_audio_s)
        await self.broadcaster.send({"type": "done", "id": utterance_id, "perf": self.last_perf})

    async def _wait_or_skip(self, awaitable):
        """Auf Rechnung oder Wiedergabe warten – oder abbrechen, wenn
        "Überspringen" gedrückt wird. Liefert (Ergebnis, übersprungen)."""
        task = asyncio.ensure_future(awaitable)
        skip_task = asyncio.create_task(self._skip.wait())
        done, _ = await asyncio.wait({task, skip_task}, return_when=asyncio.FIRST_COMPLETED)
        skip_task.cancel()
        if task in done:
            return task.result(), False
        # Eine laufende Berechnung im Thread lässt sich nicht abbrechen; ihr
        # Ergebnis wird einfach verworfen.
        task.cancel()
        return None, True

    def _perf(self, cfg, voice, totals, cached, first_audio_s) -> dict:
        audio = totals["audio"]
        return {
            "voice": (cfg["voice_labels"].get(voice)
                      or VOICES[voice]["label"].split(" – ")[0]),
            "compute_s": round(totals["compute"], 2),
            "load_s": round(totals["load"], 2),
            "audio_s": round(audio, 2),
            "cached": cached,
            "device": (self.tts.qwen_device if is_qwen(voice)
                       else "CPU" if is_pocket(voice) else None),
            "decode_s": round(totals["decode"], 2),
            "first_s": round(first_audio_s, 2) if first_audio_s is not None else None,
            "rtf": round(totals["compute"] / audio, 2) if audio else None,
            "batch": totals.get("batch", 1),
            "wait_s": round(totals["wait"], 1) if totals.get("wait") is not None else None,
        }
