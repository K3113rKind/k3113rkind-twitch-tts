/* YouTube vorlesen: deutsche Untertitel eines Videos mit der eingestellten
   Stimme vorlesen – wie Chat-Nachrichten. Die Seite verfolgt die Position
   des eingebetteten Players; erreicht das Video einen Satz, geht er in die
   Warteschlange. Die nächsten Sätze werden vorab gerechnet.
   Nutzt aus app.js: ensureAudio(), showError(), hideError() und das
   Ereignis "tts-event" (WebSocket-Nachrichten). */
(() => {
  const $ = (id) => document.getElementById(id);
  const ui = {
    url: $("yt-url"), load: $("yt-load"), info: $("yt-info"), wrap: $("yt-wrap"),
    controls: $("yt-controls"), vol: $("yt-vol"), volVal: $("yt-vol-val"),
    mode: $("yt-mode"), state: $("yt-state"),
  };
  if (!ui.load) return;

  const TICK_MS = 250;
  const PREFETCH_EVERY_MS = 2000;
  const PREFETCH_AHEAD = 4;
  const PAUSE_BACKLOG = 3;   // so viele Sätze offen -> Video anhalten
  const RESUME_BACKLOG = 1;  // weiter, wenn nur noch so viele offen sind
  const SKIP_BACKLOG = 2;    // Modus "überspringen": ab so vielen offenen nichts Neues

  const st = {
    player: null, sentences: [], next: 0, timer: null,
    lastT: 0, lastWall: 0, playing: false,
    sent: 0, done: 0, ids: new Set(), autoPaused: false,
    lastPrefetch: 0, blocked: false,
  };

  function setState(text) { ui.state.textContent = text; }

  function backlog() { return Math.max(0, st.sent - st.done); }

  async function post(url, body) {
    return fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
  }

  function resetQueue() {
    st.sent = 0; st.done = 0; st.ids.clear();
    post("/api/youtube/clear");
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
    const p = st.player;
    if (!p || typeof p.getCurrentTime !== "function") return;
    const t = p.getCurrentTime() || 0;
    const now = Date.now();
    // Gespult? (Position weicht deutlich von der erwarteten ab)
    const expected = st.playing ? st.lastT + (now - st.lastWall) / 1000 * (p.getPlaybackRate() || 1) : st.lastT;
    if (Math.abs(t - expected) > 2.5) {
      resetQueue();
      st.next = indexAt(t);
      setState("Gespult – lese ab der neuen Stelle.");
    }
    st.lastT = t; st.lastWall = now;

    if (st.playing || st.autoPaused) {
      while (st.next < st.sentences.length && st.sentences[st.next].t <= t) {
        if (ui.mode.value === "skip" && backlog() >= SKIP_BACKLOG) {
          st.next += 1;  // hinterher: diesen Satz auslassen
          continue;
        }
        send(st.sentences[st.next]);
        st.next += 1;
      }
    }
    // Hinterher? Video kurz anhalten, bis die Stimme aufgeholt hat
    if (ui.mode.value === "pause") {
      if (st.playing && !st.autoPaused && backlog() >= PAUSE_BACKLOG) {
        st.autoPaused = true;
        p.pauseVideo();
        setState("Video wartet kurz, bis die Stimme aufgeholt hat …");
      } else if (st.autoPaused && backlog() <= RESUME_BACKLOG) {
        st.autoPaused = false;
        p.playVideo();
      }
    }
    if (now - st.lastPrefetch > PREFETCH_EVERY_MS && (st.playing || st.autoPaused)) {
      st.lastPrefetch = now;
      prefetch();
    }
    if (st.playing) {
      setState(`Satz ${Math.min(st.next, st.sentences.length)} von ${st.sentences.length}` +
               (backlog() ? ` · ${backlog()} in der Warteschlange` : ""));
    }
  }

  function onStateChange(ev) {
    const S = window.YT.PlayerState;
    if (ev.data === S.PLAYING) {
      st.playing = true;
      st.lastT = st.player.getCurrentTime() || 0; st.lastWall = Date.now();
      post("/api/pause", { paused: false });
      if (!st.timer) st.timer = setInterval(tick, TICK_MS);
    } else if (ev.data === S.PAUSED) {
      st.playing = false;
      if (!st.autoPaused) {  // vom Nutzer pausiert -> Vorlesen auch anhalten
        post("/api/pause", { paused: true });
        setState("Pausiert.");
      }
    } else if (ev.data === S.ENDED) {
      st.playing = false;
      setState("Video zu Ende.");
    }
  }

  // YouTube-IFrame-API einmalig nachladen
  let apiReady = null;
  function loadApi() {
    if (apiReady) return apiReady;
    apiReady = new Promise((resolve) => {
      if (window.YT && window.YT.Player) { resolve(); return; }
      const prev = window.onYouTubeIframeAPIReady;
      window.onYouTubeIframeAPIReady = () => { if (prev) prev(); resolve(); };
      const tag = document.createElement("script");
      tag.src = "https://www.youtube.com/iframe_api";
      document.head.append(tag);
    });
    return apiReady;
  }

  function fmtDur(s) {
    if (!s) return "";
    const m = Math.floor(s / 60), sec = Math.round(s % 60);
    return ` · ${m}:${String(sec).padStart(2, "0")} min`;
  }

  ui.load.addEventListener("click", async () => {
    const url = ui.url.value.trim();
    if (!url) { ui.url.focus(); return; }
    ensureAudio();  // Nutzer-Geste: Ton im Browser freischalten
    ui.load.disabled = true;
    ui.info.classList.remove("hidden");
    ui.info.textContent = "Lade Untertitel …";
    try {
      const res = await post("/api/youtube/load", { url });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || res.statusText);
      hideError();
      st.sentences = data.sentences || [];
      st.next = 0; st.blocked = false; st.autoPaused = false;
      resetQueue();
      ui.info.textContent = `${data.title}` + (data.channel ? ` – ${data.channel}` : "") +
        fmtDur(data.duration) + ` · Original: ${data.language} · ${data.source} · ${st.sentences.length} Sätze`;
      await loadApi();
      ui.wrap.classList.remove("hidden");
      ui.controls.classList.remove("hidden");
      if (st.player) {
        st.player.loadVideoById(data.id);
        st.player.setVolume(parseInt(ui.vol.value, 10));
      } else {
        st.player = new window.YT.Player("yt-player", {
          host: "https://www.youtube-nocookie.com",
          videoId: data.id,
          playerVars: { playsinline: 1, rel: 0, cc_load_policy: 0 },
          events: {
            onReady: () => st.player.setVolume(parseInt(ui.vol.value, 10)),
            onStateChange,
          },
        });
      }
      prefetch();  // erste Sätze schon vorbereiten
      setState("Bereit – Video starten.");
    } catch (e) {
      ui.info.textContent = "";
      ui.info.classList.add("hidden");
      showError(String(e.message || e));
    } finally {
      ui.load.disabled = false;
    }
  });

  ui.vol.addEventListener("input", () => {
    ui.volVal.textContent = `${ui.vol.value} %`;
    if (st.player && st.player.setVolume) st.player.setVolume(parseInt(ui.vol.value, 10));
  });

  // Fertig gesprochene YouTube-Sätze zählen (für "hinterher?")
  window.addEventListener("tts-event", (ev) => {
    const m = ev.detail || {};
    if (m.type === "speak" && m.username === "YouTube") st.ids.add(m.id);
    else if (m.type === "done" && st.ids.has(m.id)) { st.ids.delete(m.id); st.done += 1; }
    else if (m.type === "stop_audio" || m.type === "error") {
      // abgebrochen/Fehler: offene Sätze nicht mehr mitzählen
      st.done = st.sent; st.ids.clear();
    }
  });
})();
