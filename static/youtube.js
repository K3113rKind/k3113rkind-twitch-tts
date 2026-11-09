/* YouTube vorlesen: Der Server lädt das Video (höchstens 720p), besorgt den
   Text (deutsche Untertitel, sonst Originalsprache bzw. Whisper) und
   übersetzt lokal. Danach spielt das Video hier im Browser vom eigenen
   Server; erreicht es einen Satz, geht er wie eine Chat-Nachricht in die
   Warteschlange. "Auswerfen" löscht Video und Übersetzung vom Server.
   Nutzt aus app.js: ensureAudio(), showError(), hideError() und das
   Ereignis "tts-event" (WebSocket-Nachrichten). */
(() => {
  const $ = (id) => document.getElementById(id);
  const ui = {
    url: $("yt-url"), load: $("yt-load"), eject: $("yt-eject"), info: $("yt-info"),
    wrap: $("yt-wrap"), video: $("yt-video"), controls: $("yt-controls"),
    vol: $("yt-vol"), volVal: $("yt-vol-val"), mode: $("yt-mode"), state: $("yt-state"),
  };
  if (!ui.load || !ui.video) return;

  const TICK_MS = 250;
  const PREFETCH_EVERY_MS = 2000;
  const PREFETCH_AHEAD = 4;
  const PAUSE_BACKLOG = 3;   // so viele Sätze offen -> Video anhalten
  const RESUME_BACKLOG = 1;  // weiter, wenn nur noch so viele offen sind
  const SKIP_BACKLOG = 2;    // Modus "überspringen": ab so vielen offenen nichts Neues

  const st = {
    meta: null, sentences: [], next: 0, timer: null, poll: null,
    sent: 0, done: 0, ids: new Set(), autoPaused: false, lastPrefetch: 0, blocked: false,
  };
  const v = ui.video;

  const setState = (t) => { ui.state.textContent = t; };
  const backlog = () => Math.max(0, st.sent - st.done);
  const post = (url, body) => fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}),
  });

  function resetQueue() {
    st.sent = 0; st.done = 0; st.ids.clear();
    post("/api/youtube/clear").catch(() => {});
  }

  function indexAt(t) {
    const s = st.sentences;
    let i = 0;
    while (i < s.length && s[i].t < t - 0.3) i++;
    return i;
  }

  async function send(sentence) {
    st.sent += 1;
    const res = await post("/api/youtube/say", { text: sentence.text }).catch(() => null);
    if (!res || !res.ok) {
      st.sent -= 1;
      if (res && res.status === 409 && !st.blocked) {
        st.blocked = true;
        showError("YouTube vorlesen ist gesperrt, solange der Twitch-Chat vorgelesen wird.");
      }
    }
  }

  function prefetch() {
    const texts = st.sentences.slice(st.next, st.next + PREFETCH_AHEAD).map((s) => s.text);
    if (texts.length) post("/api/youtube/prefetch", { texts }).catch(() => {});
  }

  function tick() {
    if (!st.sentences.length) return;
    const t = v.currentTime || 0;
    const running = !v.paused || st.autoPaused;
    if (running) {
      while (st.next < st.sentences.length && st.sentences[st.next].t <= t) {
        if (ui.mode.value === "skip" && backlog() >= SKIP_BACKLOG) { st.next += 1; continue; }
        send(st.sentences[st.next]);
        st.next += 1;
      }
    }
    if (ui.mode.value === "pause") {
      if (!v.paused && !st.autoPaused && backlog() >= PAUSE_BACKLOG) {
        st.autoPaused = true;
        v.pause();
        setState("Video wartet kurz, bis die Stimme aufgeholt hat …");
      } else if (st.autoPaused && backlog() <= RESUME_BACKLOG) {
        st.autoPaused = false;
        v.play().catch(() => {});
      }
    }
    const now = Date.now();
    if (running && now - st.lastPrefetch > PREFETCH_EVERY_MS) {
      st.lastPrefetch = now;
      prefetch();
    }
    if (!v.paused) {
      setState(`Satz ${Math.min(st.next, st.sentences.length)} von ${st.sentences.length}` +
               (backlog() ? ` · ${backlog()} in der Warteschlange` : ""));
    }
  }

  v.addEventListener("play", () => {
    ensureAudio();
    post("/api/pause", { paused: false }).catch(() => {});
    if (!st.timer) st.timer = setInterval(tick, TICK_MS);
  });
  v.addEventListener("pause", () => {
    if (!st.autoPaused) {  // vom Nutzer pausiert -> Vorlesen auch anhalten
      post("/api/pause", { paused: true }).catch(() => {});
      setState("Pausiert.");
    }
  });
  v.addEventListener("seeking", () => {
    resetQueue();
    st.autoPaused = false;
    st.next = indexAt(v.currentTime || 0);
    setState("Gespult – lese ab der neuen Stelle.");
  });
  v.addEventListener("ended", () => setState("Video zu Ende."));

  function fmtDur(s) {
    if (!s) return "";
    const m = Math.floor(s / 60), sec = Math.round(s % 60);
    return ` · ${m}:${String(sec).padStart(2, "0")} min`;
  }

  function showVideo(meta) {
    st.meta = meta;
    st.sentences = meta.sentences || [];
    st.next = 0; st.autoPaused = false; st.blocked = false;
    resetQueue();
    ui.info.classList.remove("hidden");
    ui.info.textContent = `${meta.title}` + (meta.channel ? ` – ${meta.channel}` : "") +
      fmtDur(meta.duration) + ` · Original: ${meta.language} · ${meta.source} · ${st.sentences.length} Sätze`;
    v.src = `/api/youtube/video?id=${encodeURIComponent(meta.id || "")}`;
    v.volume = parseInt(ui.vol.value, 10) / 100;
    ui.wrap.classList.remove("hidden");
    ui.controls.classList.remove("hidden");
    ui.eject.classList.remove("hidden");
    setState("Bereit – Video starten.");
    prefetch();  // erste Sätze schon vorbereiten
  }

  function hideVideo() {
    v.pause();
    v.removeAttribute("src");
    v.load();
    st.meta = null; st.sentences = []; st.next = 0;
    ui.wrap.classList.add("hidden");
    ui.controls.classList.add("hidden");
    ui.eject.classList.add("hidden");
    ui.info.classList.add("hidden");
    ui.info.textContent = "";
  }

  async function pollJob() {
    const res = await fetch("/api/youtube/job").catch(() => null);
    if (!res || !res.ok) return;
    const job = await res.json();
    if (job.running) {
      ui.load.disabled = true;
      ui.eject.classList.remove("hidden");
      ui.info.classList.remove("hidden");
      ui.info.textContent = (job.log && job.log.length ? job.log[job.log.length - 1] : "Starte …");
      return;
    }
    clearInterval(st.poll); st.poll = null;
    ui.load.disabled = false;
    if (job.error) {
      ui.info.classList.add("hidden");
      ui.eject.classList.add("hidden");
      showError(job.error);
    } else if (job.meta && (!st.meta || st.meta.id !== job.meta.id)) {
      hideError();
      showVideo(job.meta);
    }
  }

  ui.load.addEventListener("click", async () => {
    const url = ui.url.value.trim();
    if (!url) { ui.url.focus(); return; }
    ensureAudio();  // Nutzer-Geste: Ton im Browser freischalten
    hideVideo();
    ui.load.disabled = true;
    const res = await post("/api/youtube/prepare", { url }).catch(() => null);
    if (!res || !res.ok) {
      const body = res ? await res.json().catch(() => ({})) : {};
      showError(body.detail || "Laden fehlgeschlagen.");
      ui.load.disabled = false;
      return;
    }
    hideError();
    if (!st.poll) st.poll = setInterval(pollJob, 1000);
    pollJob();
  });

  ui.eject.addEventListener("click", async () => {
    hideVideo();
    await post("/api/youtube/eject").catch(() => {});
    clearInterval(st.poll); st.poll = null;
    ui.load.disabled = false;
    setState("");
  });

  ui.vol.addEventListener("input", () => {
    ui.volVal.textContent = `${ui.vol.value} %`;
    v.volume = parseInt(ui.vol.value, 10) / 100;
  });

  // Fertig gesprochene YouTube-Sätze zählen (für "hinterher?")
  window.addEventListener("tts-event", (ev) => {
    const m = ev.detail || {};
    if (m.type === "speak" && m.username === "YouTube") st.ids.add(m.id);
    else if (m.type === "done" && st.ids.has(m.id)) { st.ids.delete(m.id); st.done += 1; }
    else if (m.type === "stop_audio" || m.type === "error") { st.done = st.sent; st.ids.clear(); }
  });

  // Beim Öffnen der Seite: läuft eine Vorbereitung oder liegt schon ein Video bereit?
  st.poll = setInterval(pollJob, 1000);
  pollJob();
})();
