"use strict";

const $ = (id) => document.getElementById(id);
const el = {
  channel: $("channel"), voice: $("voice"),
  volume: $("volume"), volumeVal: $("volume-val"),
  speed: $("speed"), speedVal: $("speed-val"), speedNote: $("speed-note"),
  readUser: $("read_username"), readEmotes: $("read_emotes"),
  usernameStyle: $("username_style"),
  readMentions: $("read_mentions"), readSmileys: $("read_smileys"),
  keepAwake: $("keep_speakers_awake"), loudness: $("loudness_norm"),
  chatLlm: $("chat_llm"), poolChat: $("voice_pool_chat"), poolYt: $("voice_pool_youtube"), pool: $("voice-pool"),
  keepaliveLevel: $("keepalive_level"), keepaliveVal: $("keepalive-val"),
  cooldown: $("cooldown_seconds"), queueLimit: $("queue_limit"),
  blocklist: $("bot_blocklist"),
  skip: $("skip"), pause: $("pause"), startstop: $("startstop"), audioUnlock: $("audio-unlock"),
  applyChannel: $("apply-channel"), applyVoice: $("apply-voice"), renameVoice: $("rename-voice"),
  hideVoice: $("hide-voice"), manageVoices: $("manage-voices"), voiceManager: $("voice-manager"),
  voiceFilter: $("voice-filter"), voiceGroups: $("voice-groups"),
  prodUrl: $("prod-url"), shareToken: $("share-token"), shareSelfUrl: $("share-self-url"),
  shareStatus: $("share-status"), sharePublish: $("share-publish"),
  dot: $("dot"), statusText: $("status-text"), now: $("now-playing"),
  errorBanner: $("error-banner"),
  sayText: $("say-text"), sayBtn: $("say"), sayCancel: $("say-cancel"), sayHint: $("say-hint"),
  sayDelayed: $("say-delayed"), sayPrebuffer: $("say-prebuffer"),
  cloneUrl: $("clone-url"), cloneFile: $("clone-file"), cloneName: $("clone-name"),
  cloneSeconds: $("clone-seconds"), cloneStart: $("clone-start"), cloneStatus: $("clone-status"),
  cloneList: $("clone-list"), cloneResult: $("clone-result"),
  cloneCandsBox: $("clone-cands-box"), cloneCands: $("clone-cands"), cloneText: $("clone-text"), cloneTextSave: $("clone-text-save"),
  tempo: $("qwen_tempo"), tempoRow: $("tempo-row"),
  style: $("qwen_style"), styleCustom: $("qwen_style_custom"), styleRow: $("style-row"),
  temperature: $("qwen_temperature"), temperatureVal: $("temperature-val"), temperatureRow: $("temperature-row"),
  perf: $("perf"),
};

/* --------------------------------------------------------- Audio-Player */
// Der Ton kommt als durchgehender MP3-Stream vom Server (/stream) und läuft
// über ein normales <audio>-Element – wie ein Internetradio. Nur so spielt
// das iPhone auch im Hintergrund und bei gesperrtem Display weiter: iOS
// pausiert reines WebAudio samt JavaScript, ein laufendes Media-Element
// dagegen nicht. Hintergrundton und Weckimpuls für die Lautsprecher mischt
// der Server direkt in den Stream, Lautstärke ebenfalls (iOS ignoriert
// audio.volume).
const player = new Audio();
player.preload = "none";
player.setAttribute("playsinline", "");
let wantPlay = false;     // Nutzer hat den Ton auf diesem Gerät eingeschaltet
let restartTimer = null;

// Die Chat-Verbindung ist serverseitig global, die Audiofreigabe aber pro
// Gerät: Sie braucht eine Nutzer-Geste auf genau diesem Gerät
// (Autoplay-Sperre). Öffnet man die Seite auf einem zweiten Gerät, während
// die Verbindung schon läuft, gibt es dafür diesen eigenen Knopf.
function updateUnlockButton() {
  el.audioUnlock.classList.toggle("hidden", !player.paused);
}

// Immer frisch verbinden statt fortsetzen: Nach einer Pause stünde sonst
// alter Ton im Puffer.
function startStream() {
  wantPlay = true;
  clearTimeout(restartTimer);
  player.src = "/stream?t=" + Date.now();
  const p = player.play();
  if (p) p.catch((e) => { console.warn("Wiedergabe blockiert", e); updateUnlockButton(); });
  setMediaSession();
}

function stopStream() {
  wantPlay = false;
  clearTimeout(restartTimer);
  player.pause();
  player.removeAttribute("src");
  player.load();  // Verbindung zum Server wirklich schließen
  updateUnlockButton();
}

// Verbindungsabbruch (WLAN-Wechsel, Server-Neustart …): neu verbinden.
function scheduleRestart() {
  if (!wantPlay) return;
  clearTimeout(restartTimer);
  restartTimer = setTimeout(startStream, 2000);
}

function ensureAudio() {
  if (player.paused) startStream();
}

player.addEventListener("playing", updateUnlockButton);
player.addEventListener("pause", updateUnlockButton);
player.addEventListener("error", scheduleRestart);
player.addEventListener("ended", scheduleRestart);
player.addEventListener("stalled", scheduleRestart);

// Beim Zurückholen der Seite (z. B. nach einem Anruf, der die Wiedergabe
// unterbrochen hat) wieder anwerfen.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && wantPlay && player.paused) startStream();
});

// Sperrbildschirm/Kontrollzentrum: Titel anzeigen, Play/Pause bedienbar.
function setMediaSession() {
  if (!("mediaSession" in navigator)) return;
  const ch = el.channel.value ? "#" + el.channel.value : "Twitch-Chat";
  navigator.mediaSession.metadata = new MediaMetadata({
    title: "Chat wird vorgelesen", artist: ch, album: "K3113rkind's Twitch TTS",
    artwork: [{ src: "/static/favicon.svg", sizes: "any", type: "image/svg+xml" }],
  });
  navigator.mediaSession.setActionHandler("play", startStream);
  navigator.mediaSession.setActionHandler("pause", stopStream);
  navigator.mediaSession.setActionHandler("stop", stopStream);
}

el.audioUnlock.addEventListener("click", startStream);
updateUnlockButton();

// Während die Sprache berechnet wird: Hinweis mit mitlaufenden Sekunden.
// Bei langen Texten zusätzlich: wie viele Sätze fertig sind und ab wann es
// losgeht (Vorpuffer), während der Wiedergabe: welcher Satz gerade spielt.
let genTimer = null;
let genRender = null;
let progress = null;
function progressText() {
  const p = progress;
  if (!p || p.total < 2) return "";
  if (!p.playing) {
    return `${p.done} von ${p.total} Abschnitten fertig` +
      (p.needed > 1 ? (p.needed >= p.total ? " · Start, sobald alles fertig ist"
                                           : ` · Start, sobald ${p.needed} fertig sind`) : "") +
      (p.workers > 1 ? ` · ${p.workers} Prozesse` : "");
  }
  return `Abschnitt ${p.playing} von ${p.total} · ${p.done} fertig` +
    (p.done <= p.playing && p.playing < p.total ? " · nächster rechnet noch" : "");
}
/* ------------------------------------------------- Statusfeld (3 Zeilen)
   Oben die aktuelle Aktion (wird laufend aktualisiert), darunter die zwei
   davor; ältere verschwinden. */
const statusLines = [];  // [{key, text, time}]
function statusSet(key, text) {
  const now = new Date().toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  if (statusLines.length && statusLines[0].key === key) {
    statusLines[0].text = text;
  } else {
    statusLines.unshift({ key, text, time: now });
    statusLines.length = Math.min(statusLines.length, 3);
  }
  const box = document.getElementById("status-log");
  if (!box) return;
  box.innerHTML = "";
  for (const l of statusLines) {
    const div = document.createElement("div");
    div.className = "line";
    const t = document.createElement("span");
    t.className = "time";
    t.textContent = l.time;
    div.append(t, document.createTextNode(l.text));
    box.append(div);
  }
}
let actionSeq = 0;  // jede neue Nachricht = neue Zeile

let speakingMsg = null;
function renderNowProgress() {
  if (!speakingMsg) return;
  const pt = progressText();
  statusSet("speak" + actionSeq, `Spricht: ${speakingMsg.username}` + (pt ? ` · ${pt}` : ""));
}
function showGenerating(msg) {
  stopGenerating();
  progress = null;
  const started = msg.ts || Date.now();  // nach Tab neu öffnen: echte Startzeit vom Server
  actionSeq += 1;
  const key = "gen" + actionSeq;
  speakingMsg = null;
  // Text der Nachricht im Textfeld, Status im Statusfeld darüber
  el.now.innerHTML = "";
  const user = document.createElement("span");
  user.className = "user";
  user.textContent = msg.username;
  el.now.append(user, document.createTextNode(": " + msg.text));
  const render = () => {
    const s = Math.floor((Date.now() - started) / 1000);
    const pt = progressText();
    statusSet(key, `Erzeuge Sprache für ${msg.username} … ${s} s` + (pt ? ` · ${pt}` : ""));
  };
  render();
  genRender = render;
  genTimer = setInterval(render, 1000);
}
function stopGenerating() {
  clearInterval(genTimer);
  genTimer = null;
  genRender = null;
}

function showNow(msg) {
  el.now.innerHTML = "";
  const user = document.createElement("span");
  user.className = "user";
  user.textContent = msg.username;
  el.now.append(user, document.createTextNode(": " + msg.text));
  speakingMsg = msg;
  renderNowProgress();
}

/* ------------------------------------------------------------ WebSocket */
let ws = null;
let wsRetry = 1000;
let pingTimer = null;

function sendWs(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => {
    wsRetry = 1000;
    refreshStatus();
    // Regelmäßiges Lebenszeichen: Reverse-Proxys (z. B. Traefik/Pangolin)
    // kappen sonst Verbindungen, über die längere Zeit nichts läuft.
    clearInterval(pingTimer);
    pingTimer = setInterval(() => sendWs({ type: "ping" }), 25000);
  };
  ws.onclose = () => {
    clearInterval(pingTimer);
    setTimeout(connectWs, wsRetry);
    wsRetry = Math.min(wsRetry * 2, 15000);
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    // für Zusatzmodule (YouTube vorlesen)
    window.dispatchEvent(new CustomEvent("tts-event", { detail: msg }));
    switch (msg.type) {
      case "generating":
        showGenerating(msg);
        break;
      case "progress":
        progress = msg;
        if (genRender) genRender(); else renderNowProgress();
        break;
      case "speak":
        stopGenerating();
        showPerf(msg.perf);
        speaking = true;
        updateSkipButton();
        showNow(msg);
        break;
      case "done":
        if (msg.perf) showPerf(msg.perf);
        if (speakingMsg) statusSet("done" + actionSeq, `Fertig: ${speakingMsg.username}`);
        speakingMsg = null;
        // fällt durch
      case "stop_audio":
        if (msg.type === "stop_audio" && (speakingMsg || genTimer)) statusSet("stop" + actionSeq, "Abgebrochen");
        speakingMsg = null;
        stopGenerating();
        speaking = false;
        updateSkipButton();
        el.now.textContent = "–";
        progress = null;
        endTestText();
        break;
      case "status":
        applyStatus(msg);
        break;
      case "paused":
        if (msg.paused !== paused) { actionSeq += 1; statusSet("pause" + actionSeq, msg.paused ? "Pausiert" : "Fortgesetzt"); }
        setPaused(msg.paused);
        break;
      case "voices_changed":
        refreshStatus();
        loadCloneList();
        if (!el.voiceManager.classList.contains("hidden")) loadVoiceCatalog();
        break;
      case "error":
        statusSet("err" + actionSeq, "Fehler: " + (msg.message || ""));
        speakingMsg = null;
        stopGenerating();
        el.now.textContent = "–";
        showError(msg.message);
        endTestText();
        break;
    }
  };
}

/* ---------------------------------------------------------------- Status */
// Überspringen geht auch ohne laufenden Chat, z. B. bei einem Testtext.
let running = false;
let speaking = false;
function updateSkipButton() { el.skip.disabled = !running && !speaking; }

let paused = false;
function setPaused(p) {
  paused = !!p;
  el.pause.textContent = paused ? "▶ Weiter" : "⏸ Pause";
  el.pause.classList.toggle("running", paused);
  el.pause.title = paused
    ? "Vorlesen fortsetzen – es geht an derselben Stelle weiter"
    : "Vorlesen anhalten/fortsetzen – Chat und Testtext";
}

function applyStatus(s) {
  el.dot.className = "dot " + (s.connected ? "ok" : s.running ? "warn" : "");
  el.statusText.textContent = s.connected
    ? `Verbunden mit #${s.channel}`
    : s.running ? "Verbinde …" : "Nicht verbunden";
  if (s.warming) el.statusText.textContent += " · Stimme wird aufgewärmt …";
  el.startstop.textContent = s.running ? "Vorlesen stoppen" : "Vorlesen starten";
  el.startstop.classList.toggle("running", s.running);
  running = s.running;
  if (typeof s.paused === "boolean") setPaused(s.paused);
  // Rückfall: "fertig" über die WebSocket verpasst (z. B. Verbindung kurz weg)
  if (testActive && s.test_active === false && Date.now() - testSince > 4000) endTestText();
  // Tab neu geöffnet, während ein Testtext läuft: Knöpfe wieder sperren
  if (!testActive && s.test_active === true) { testActive = true; sayBusy = true; testSince = Date.now(); }
  updateSkipButton();
  updateSayButton();
  fillVoices(s.voices || []);
  if (s.workers) fillWorkers(s.workers);
  if (s.last_perf && !el.perf.textContent) showPerf(s.last_perf);
  if (s.error) showError(s.error); else hideError();
}

function showError(text) {
  el.errorBanner.textContent = text;
  el.errorBanner.classList.remove("hidden");
}
function hideError() { el.errorBanner.classList.add("hidden"); }

async function refreshStatus() {
  applyStatus(await fetch("/api/status").then((r) => r.json()));
}

function fillVoices(voices) {
  const selected = el.voice.dataset.selected || el.voice.value;
  el.voice.innerHTML = "";
  for (const v of voices) {
    const opt = document.createElement("option");
    opt.value = v.key;
    opt.textContent = v.label;
    opt.dataset.defaultLabel = v.default_label || v.label;
    opt.dataset.qwen = v.qwen ? "1" : "";
    opt.dataset.styleable = v.styleable ? "1" : "";
    opt.dataset.pocket = v.pocket ? "1" : "";
    el.voice.append(opt);
  }
  if (voices.some((v) => v.key === selected)) el.voice.value = selected;
  updateTemperatureRow();
  renderVoicePool(voices);
}

// Stimmen-Wechsel: Auswahl aus Kokoro- und Pocket-Stimmen
let poolSelected = [];
function renderVoicePool(voices) {
  // Aufklappliste mit Häkchen (mehrere Stimmen wählbar)
  if (!el.pool) return;
  const list = voices.filter((v) => !v.qwen);
  const keys = list.map((v) => v.key).join(",");
  if (el.pool.dataset.keys === keys) { updatePoolButton(); return; }
  el.pool.dataset.keys = keys;
  el.pool.innerHTML = "";
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "multi-btn";
  const panel = document.createElement("div");
  panel.className = "multi-panel hidden";
  for (const v of list) {
    const label = document.createElement("label");
    label.className = "check";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = v.key;
    cb.checked = poolSelected.includes(v.key);
    cb.addEventListener("change", () => {
      poolSelected = [...panel.querySelectorAll("input:checked")].map((x) => x.value);
      updatePoolButton();
      saveConfig();
    });
    label.append(cb, document.createTextNode(" " + v.label));
    panel.append(label);
  }
  btn.addEventListener("click", (ev) => { ev.stopPropagation(); panel.classList.toggle("hidden"); });
  panel.addEventListener("click", (ev) => ev.stopPropagation());
  document.addEventListener("click", () => panel.classList.add("hidden"));
  el.pool.append(btn, panel);
  updatePoolButton();
}
function updatePoolButton() {
  const btn = el.pool && el.pool.querySelector(".multi-btn");
  if (!btn) return;
  const labels = [...el.pool.querySelectorAll("input:checked")].map((cb) => cb.parentElement.textContent.trim());
  btn.textContent = (labels.length === 0 ? "Stimmen auswählen …"
    : labels.length <= 2 ? labels.join(", ") : `${labels.length} Stimmen ausgewählt`) + "  ▾";
}

// Betonungs-Regler nur bei Qwen-Stimmen zeigen
function updateTemperatureRow() {
  const opt = el.voice.selectedOptions[0];
  el.temperatureRow.classList.toggle("hidden", !(opt && opt.dataset.qwen));
  el.styleRow.classList.toggle("hidden", !(opt && opt.dataset.styleable));
  el.styleCustom.classList.toggle("hidden", el.style.value !== "eigen");
  // Qwen: Presets grob per Anweisung, eigene Stimmen werden nachträglich gestreckt
  const qwen = !!(opt && opt.dataset.qwen), styleable = !!(opt && opt.dataset.styleable);
  const pocket = !!(opt && opt.dataset.pocket);
  el.speed.disabled = pocket;
  el.tempoRow.classList.toggle("hidden", !(qwen && !styleable));
  el.speedNote.classList.toggle("hidden", !qwen && !pocket);
  el.speedNote.textContent = pocket
    ? "Pocket TTS: kein Tempo-Regler, Tempo kommt aus der Stimme"
    : !qwen ? "" : styleable
    ? "Qwen: wirkt grob als Anweisung (langsamer/schneller), nicht exakt"
    : "Eigene Stimme: Tempo wird nachträglich angepasst – Verfahren unten wählbar";
}

// Leistung der letzten Ausgabe (Testtext oder Twitch), z. B.
// "Letzte Ausgabe: 7,4 s Rechenzeit für 2,1 s Audio · RTF 3,52 · Qwen 1.7B · Vivian"
const fmt = (n) => Number(n).toLocaleString("de-DE", { maximumFractionDigits: 2, minimumFractionDigits: 1 });
function showPerf(p) {
  if (!p) return;
  if (p.cached) {
    el.perf.innerHTML = `Letzte Ausgabe: <b>aus dem Zwischenspeicher</b> (sofort) · ${fmt(p.audio_s)} s Audio`;
    el.perf.append(" · " + p.voice);
    return;
  }
  let text = `Letzte Ausgabe: <b>${fmt(p.compute_s)} s</b> Rechenzeit für ${fmt(p.audio_s)} s Audio`;
  if (p.rtf != null) text += ` · RTF ${fmt(p.rtf)}`;
  if (p.decode_s >= 0.1) text += ` · davon Klang-Decoder ${fmt(p.decode_s)} s`;
  if (p.first_s != null && p.audio_s > 0 && p.compute_s - p.first_s > 0.5) text += ` · erster Ton nach ${fmt(p.first_s)} s`;
  if (p.load_s >= 1) text += ` · + ${fmt(p.load_s)} s Modell laden`;
  if (p.wait_s != null) text += ` · <b>${fmt(p.wait_s)} s</b> nach Eingang`;
  el.perf.innerHTML = text;
  el.perf.append(" · " + p.voice + (p.device ? ` · ${p.device}` : ""));
}

/* ------------------------------------------------------- Konfiguration */
let saveTimer = null;

function collectConfig() {
  return {
    channel: cleanChannel(el.channel.value),
    voice: el.voice.value,
    volume: parseFloat(el.volume.value),
    speed: parseFloat(el.speed.value),
    read_username: el.readUser.checked,
    username_style: el.usernameStyle.value,
    read_emotes: el.readEmotes.checked,
    read_mentions: el.readMentions.checked,
    read_smileys: el.readSmileys.checked,
    keep_speakers_awake: el.keepAwake.checked,
    loudness_norm: el.loudness.checked,
    voice_pool_chat: el.poolChat.checked,
    chat_llm: el.chatLlm.checked,
    voice_pool_youtube: el.poolYt.checked,
    voice_pool: poolSelected,
    keepalive_level: parseFloat(el.keepaliveLevel.value),
    cooldown_seconds: parseInt(el.cooldown.value || "0", 10),
    queue_limit: parseInt(el.queueLimit.value || "1", 10),
    bot_blocklist: el.blocklist.value.split("\n").map((s) => s.trim()).filter(Boolean),
    qwen_temperature: parseFloat(el.temperature.value),
    qwen_style: el.style.value,
    qwen_tempo: el.tempo.value,
    qwen_style_custom: el.styleCustom.value,
  };
}

function saveConfig() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectConfig()),
    });
  }, 300);
}

function applyConfig(c) {
  fillShareInfo(c);
  el.channel.value = c.channel;
  el.voice.dataset.selected = c.voice;
  el.volume.value = c.volume;
  el.speed.value = c.speed;
  el.readUser.checked = c.read_username;
  el.usernameStyle.value = c.username_style;
  el.readEmotes.checked = c.read_emotes;
  el.readMentions.checked = c.read_mentions;
  el.readSmileys.checked = c.read_smileys;
  el.keepAwake.checked = c.keep_speakers_awake;
  el.loudness.checked = c.loudness_norm !== false;
  el.poolChat.checked = !!c.voice_pool_chat;
  el.chatLlm.checked = !!c.chat_llm;
  el.poolYt.checked = !!c.voice_pool_youtube;
  poolSelected = Array.isArray(c.voice_pool) ? c.voice_pool : [];
  if (el.pool) el.pool.querySelectorAll("input").forEach((cb) => { cb.checked = poolSelected.includes(cb.value); });
  updatePoolButton();
  el.keepaliveLevel.value = c.keepalive_level;
  el.cooldown.value = c.cooldown_seconds;
  el.queueLimit.value = c.queue_limit;
  el.blocklist.value = c.bot_blocklist.join("\n");
  voiceLabels = c.voice_labels || {};
  el.temperature.value = c.qwen_temperature ?? 0.9;
  el.style.value = c.qwen_style || "neutral";
  el.tempo.value = c.qwen_tempo || "auto";
  el.styleCustom.value = c.qwen_style_custom || "";
  updateTemperatureRow();
  updateSliderLabels();
  updateKeepaliveLabel();
}

function updateKeepaliveLabel() {
  const v = parseFloat(el.keepaliveLevel.value);
  el.keepaliveVal.textContent = v === 0 ? "aus" : v.toFixed(3);
}

el.keepaliveLevel.addEventListener("input", () => { updateKeepaliveLabel(); saveConfig(); });

function updateSliderLabels() {
  el.volumeVal.textContent = Math.round(parseFloat(el.volume.value) * 100) + " %";
  el.speedVal.textContent = parseFloat(el.speed.value).toFixed(2) + "\u00d7";
  el.temperatureVal.textContent = parseFloat(el.temperature.value).toFixed(2);
}

/* ---------------------------------------------------------------- Events */
for (const input of [el.channel, el.voice, el.readUser, el.usernameStyle, el.readEmotes,
                     el.readMentions, el.readSmileys, el.keepAwake, el.loudness, el.chatLlm, el.poolChat, el.poolYt, el.cooldown, el.queueLimit, el.blocklist]) {
  input.addEventListener("change", saveConfig);
}
el.voice.addEventListener("change", () => {
  el.voice.dataset.selected = el.voice.value;
  updateTemperatureRow();
  // Qwen-/Pocket-Stimme schon beim Auswählen vorladen, nicht erst bei "Wechseln"
  const opt = el.voice.selectedOptions[0];
  if (opt && (opt.dataset.qwen || opt.dataset.pocket)) {
    fetch("/api/warmup", { method: "POST", headers: { "Content-Type": "application/json" },
                           body: JSON.stringify({ voice: el.voice.value }) }).catch(() => {});
  }
});
el.style.addEventListener("change", () => { updateTemperatureRow(); saveConfig(); });
el.tempo.addEventListener("change", saveConfig);
el.styleCustom.addEventListener("change", saveConfig);

// Ganzen Twitch-Link einfügen können: https://www.twitch.tv/name -> name
function cleanChannel(v) {
  v = (v || "").trim();
  const m = v.match(/twitch\.tv\/(?:(?:popout|moderator|embed)\/)?([A-Za-z0-9_]{2,25})/i);
  if (m) v = m[1];
  return v.replace(/^[@#]+/, "").trim().toLowerCase();
}
el.channel.addEventListener("input", () => {
  if (/twitch\.tv\/[A-Za-z0-9_]{2,}|^[@#]/i.test(el.channel.value)) el.channel.value = cleanChannel(el.channel.value);
});
el.channel.addEventListener("change", setMediaSession);
for (const slider of [el.volume, el.speed, el.temperature]) {
  slider.addEventListener("input", () => { updateSliderLabels(); saveConfig(); });
}

el.skip.addEventListener("click", () => fetch("/api/skip", { method: "POST" }));
document.getElementById("emergency").addEventListener("click", async () => {
  if (!confirm("Not-Aus: Alles Laufende sofort abbrechen (YouTube-Vertonung, Testtexte, " +
               "Warteschlange) und die Rechenprozesse neu starten?")) return;
  const res = await fetch("/api/emergency_stop", { method: "POST" }).catch(() => null);
  if (res && res.ok) {
    hideError();
    testActive = false; sayBusy = false; updateSayButton();
    window.dispatchEvent(new CustomEvent("tts-emergency"));
  } else {
    showError("Not-Aus fehlgeschlagen – Container neu starten.");
  }
});
el.pause.addEventListener("click", async () => {
  const res = await fetch("/api/pause", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paused: !paused }),
  }).catch(() => null);
  if (res && res.ok) setPaused((await res.json()).paused);
});

/* Übernehmen-Knöpfe: erst die aktuelle Eingabe speichern (das normale
   change-Event könnte noch nicht gefeuert haben, wenn direkt aus dem Feld
   heraus geklickt wird), dann serverseitig anwenden. */
async function applyNow(button, endpoint) {
  ensureAudio(); // Nutzer-Geste mitnehmen: löst ggf. die Autoplay-Sperre
  button.disabled = true;
  const original = button.textContent;
  button.textContent = "…";
  try {
    clearTimeout(saveTimer);
    await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectConfig()),
    });
    const res = await fetch(endpoint, { method: "POST" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      showError(body.detail || res.statusText);
    } else {
      hideError();
      applyStatus(await res.json());
    }
  } catch (e) {
    showError("Wechsel fehlgeschlagen: " + e);
  } finally {
    button.textContent = original;
    button.disabled = false;
    refreshStatus();
  }
}

el.applyChannel.addEventListener("click", () => applyNow(el.applyChannel, "/api/apply/channel"));
el.applyVoice.addEventListener("click", () => applyNow(el.applyVoice, "/api/apply/voice"));

/* Stimme umbenennen: eigener Anzeigename, leer = Standardname */
let voiceLabels = {};
el.renameVoice.addEventListener("click", async () => {
  const opt = el.voice.selectedOptions[0];
  if (!opt) return;
  const name = window.prompt(
    `Neuer Name für diese Stimme (leer lassen = Standard):\n${opt.dataset.defaultLabel}`,
    opt.textContent,
  );
  if (name === null) return; // abgebrochen
  const labels = { ...voiceLabels };
  if (name.trim() && name.trim() !== opt.dataset.defaultLabel) labels[opt.value] = name.trim();
  else delete labels[opt.value];
  const res = await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ voice_labels: labels }),
  });
  if (res.ok) {
    voiceLabels = (await res.json()).voice_labels || {};
    refreshStatus();
  } else {
    showError("Umbenennen fehlgeschlagen");
  }
});

// Enter im Kanalfeld verhält sich wie der Wechseln-Knopf.
el.channel.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") {
    ev.preventDefault();
    el.applyChannel.click();
  }
});

el.startstop.addEventListener("click", async () => {
  ensureAudio(); // Nutzer-Geste: löst die Autoplay-Sperre des Browsers
  el.startstop.disabled = true;
  const running = el.startstop.classList.contains("running");
  try {
    const res = await fetch(running ? "/api/stop" : "/api/start", { method: "POST" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      showError(body.detail || res.statusText);
    } else {
      hideError();
      applyStatus(await res.json());
    }
  } finally {
    el.startstop.disabled = false;
    refreshStatus();
  }
});

/* ------------------------------------------------------------- Testtext */
// Testtext nur, wenn der Chat nicht vorgelesen wird
let sayBusy = false;
let testActive = false;  // Testtext wird erzeugt/gespielt -> Knopf gesperrt
let testSince = 0;
function endTestText() {
  if (!testActive) return;
  testActive = false;
  sayBusy = false;
  updateSayButton();
}
function updateSayButton() {
  el.sayBtn.disabled = running || sayBusy;
  el.sayDelayed.disabled = running || sayBusy;
  el.sayText.disabled = running;
  el.sayHint.classList.toggle("hidden", !running);
}

el.sayBtn.addEventListener("click", () => sayTest("direct"));
el.sayDelayed.addEventListener("click", () => sayTest(el.sayPrebuffer.value));

async function sayTest(prebuffer) {
  const text = el.sayText.value.trim();
  if (!text) { el.sayText.focus(); return; }
  if (running) return;
  ensureAudio(); // Nutzer-Geste: löst die Autoplay-Sperre des Browsers
  sayBusy = true;
  // schon vor der Anfrage: ein sehr schnelles "fertig"/"Fehler" vom Server
  // käme sonst vor der Antwort an und der Knopf bliebe gesperrt
  testActive = true;
  testSince = Date.now();
  updateSayButton();
  try {
    // Aktuelle Stimme/Geschwindigkeit sofort speichern, nicht erst nach 300 ms
    clearTimeout(saveTimer);
    await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectConfig()),
    });
    const res = await fetch("/api/say", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, prebuffer }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      showError(body.detail || res.statusText);
      sayBusy = false;
      testActive = false;
    } else {
      hideError();
      // gesperrt bis "fertig", "abgebrochen" oder Fehler; sofort Rückmeldung, bis der Server "Erzeuge Sprache" meldet
      if (!genTimer && !speaking) showGenerating({ username: "Test", text });
    }
  } catch (e) {
    showError("Vorlesen fehlgeschlagen: " + e);
    sayBusy = false;
    testActive = false;
  } finally {
    updateSayButton();
  }
}

el.sayCancel.addEventListener("click", () => fetch("/api/say/cancel", { method: "POST" }));

// Hörprobe: fester Satz, gleiche Sperre wie beim Testtext
const previewBtn = document.getElementById("preview-voice");
previewBtn.addEventListener("click", async () => {
  if (running) { showError("Hörprobe geht nur, solange der Chat nicht vorgelesen wird."); return; }
  ensureAudio();
  previewBtn.disabled = true;
  try {
    clearTimeout(saveTimer);
    await fetch("/api/config", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(collectConfig()),
    });
    const res = await fetch("/api/preview", { method: "POST" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      showError(body.detail || res.statusText);
    } else hideError();
  } catch (e) {
    showError("Hörprobe fehlgeschlagen: " + e);
  } finally {
    previewBtn.disabled = false;
  }
});

// Strg+Enter (bzw. Cmd+Enter) im Textfeld = Vorlesen
el.sayText.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) {
    ev.preventDefault();
    sayTest("direct");
  }
});

/* ------------------------------------------------ Eigene Stimme erstellen */
let clonePoll = null;

function showCloneStatus(job) {
  const lines = [...(job.log || [])];
  if (job.error) lines.push("Fehler: " + job.error);
  el.cloneStatus.textContent = lines.join("\n");
  el.cloneStatus.classList.toggle("hidden", lines.length === 0);
  el.cloneStatus.classList.toggle("err", !!job.error);
  el.cloneStart.disabled = !!job.running;
  el.cloneStart.textContent = job.running ? `Erstelle „${job.name}“ …` : "Stimme erstellen";
  if (job.result) {
    const r = job.result;
    el.cloneStatus.textContent += `\nVerwendet: ${r.start}–${r.end} s (${r.seconds} s) aus der Aufnahme.`;
    if (el.cloneResult.dataset.name !== r.name) {
      el.cloneText.value = r.text;
      el.cloneResult.dataset.name = r.name;
      loadCandidates(r.name);
    }
    el.cloneResult.classList.remove("hidden");
  }
}

async function pollClone() {
  const job = await fetch("/api/clone").then((r) => r.json()).catch(() => null);
  if (!job) return;
  showCloneStatus(job);
  if (!job.running) {
    clearInterval(clonePoll);
    clonePoll = null;
    refreshStatus();
  }
}

async function startClone(overwrite = false) {
  const name = el.cloneName.value.trim().toLowerCase();
  const url = el.cloneUrl.value.trim();
  const file = el.cloneFile.files[0];
  if (!name) { showError("Bitte einen Namen für die Stimme eingeben."); el.cloneName.focus(); return; }
  if (!url && !file) { showError("Bitte einen Video-Link eingeben oder eine Datei auswählen."); return; }
  if (url && file) { showError("Bitte entweder Link oder Datei – nicht beides."); return; }
  hideError();
  const form = new FormData();
  form.append("name", name);
  form.append("seconds", el.cloneSeconds.value || "20");
  form.append("overwrite", overwrite ? "true" : "false");
  if (url) form.append("url", url); else form.append("file", file);
  el.cloneStart.disabled = true;
  el.cloneResult.classList.add("hidden");
  el.cloneResult.dataset.name = "";
  const res = await fetch("/api/clone", { method: "POST", body: form });
  if (res.status === 409) {
    const body = await res.json().catch(() => ({}));
    if (body.detail === "EXISTS") {
      if (window.confirm(`Die Stimme „${name}“ gibt es schon. Überschreiben?`)) return startClone(true);
      el.cloneStart.disabled = false;
      return;
    }
    showError(body.detail || "Läuft bereits.");
  } else if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    showError(body.detail || res.statusText);
    el.cloneStart.disabled = false;
    return;
  }
  showCloneStatus(await res.json().catch(() => ({ running: true, name, log: [] })));
  if (!clonePoll) clonePoll = setInterval(pollClone, 1500);
}

el.cloneStart.addEventListener("click", () => startClone(false));
// ------------------------------------------- Aufnahme vom Gerätemikrofon
// 12 s aufnehmen und sofort als neue Stimme "aufnahmeNN" erstellen
// (umbenennen geht danach in der Stimmenliste). Browser erlauben das
// Mikrofon nur über HTTPS (oder localhost).
const REC_SECONDS = 12;
const recBtn = document.getElementById("clone-rec");
const recInfo = document.getElementById("clone-rec-info");

async function nextRecordingName() {
  const list = await fetch("/api/clone/voices").then((r) => r.json()).catch(() => []);
  let max = 0;
  for (const v of list || []) {
    const m = /^aufnahme(\d+)$/.exec(v.name);
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  return "aufnahme" + String(max + 1).padStart(2, "0");
}

function recorderType() {
  if (!window.MediaRecorder) return null;
  for (const t of ["audio/mp4", "audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"]) {
    if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";  // Browser-Standard
}

async function submitRecording(blob, ext) {
  const name = await nextRecordingName();
  const form = new FormData();
  form.append("name", name);
  form.append("seconds", String(REC_SECONDS - 2));  // Vorlage etwas kürzer: Anfang/Ende frei wählbar
  form.append("overwrite", "false");
  form.append("direct", "true");  // Mikrofon: keine Musik, nicht trennen
  form.append("file", blob, `aufnahme${ext}`);
  el.cloneStart.disabled = true;
  el.cloneResult.classList.add("hidden");
  el.cloneResult.dataset.name = "";
  const res = await fetch("/api/clone", { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    showError(body.detail === "EXISTS" ? `„${name}“ gibt es schon.` : (body.detail || res.statusText));
    el.cloneStart.disabled = false;
    return;
  }
  recInfo.textContent += ` Stimme „${name}“ wird erstellt …`;
  showCloneStatus(await res.json().catch(() => ({ running: true, name, log: [] })));
  if (!clonePoll) clonePoll = setInterval(pollClone, 1500);
}

recBtn.addEventListener("click", async () => {
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError("Mikrofon-Aufnahme geht im Browser nur über HTTPS. Alternative: mit der Sprachmemos-App " +
              "aufnehmen und oben unter „Datei hochladen“ auswählen.");
    return;
  }
  const type = recorderType();
  if (type === null) { showError("Dieser Browser kann nicht aufnehmen (MediaRecorder fehlt)."); return; }
  let stream;
  try {
    // Rohsignal: keine Rauschunterdrückung/Pegelautomatik – die verfälschen die Stimme
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
  } catch (e) {
    showError("Kein Zugriff aufs Mikrofon: " + (e.message || e));
    return;
  }
  hideError();
  // Pegel live messen (nur Anzeige): zu leise Aufnahmen fallen sofort auf
  let meter = null, peakDb = -100;
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const an = ctx.createAnalyser();
    an.fftSize = 2048;
    ctx.createMediaStreamSource(stream).connect(an);
    const buf = new Float32Array(an.fftSize);
    meter = { ctx, read() {
      an.getFloatTimeDomainData(buf);
      let p = 0;
      for (const v of buf) p = Math.max(p, Math.abs(v));
      const db = 20 * Math.log10(Math.max(p, 1e-6));
      peakDb = Math.max(peakDb, db);
      return db;
    } };
  } catch (e) { meter = null; }
  const rec = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
  const parts = [];
  const started = Date.now();
  rec.ondataavailable = (ev) => { if (ev.data && ev.data.size) parts.push(ev.data); };
  rec.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    if (meter) meter.ctx.close().catch(() => {});
    recBtn.disabled = false;
    recBtn.textContent = "🎙 Aufnahme (12 Sek.)";
    const mime = rec.mimeType || type || "audio/webm";
    const ext = mime.includes("mp4") ? ".m4a" : mime.includes("ogg") ? ".ogg" : ".webm";
    const blob = new Blob(parts, { type: mime });
    const secs = ((Date.now() - started) / 1000).toFixed(1).replace(".", ",");
    const level = meter ? ` · Spitze ${Math.round(peakDb)} dB` : "";
    if (meter && peakDb < -30) {
      showError(`Aufnahme sehr leise (Spitze ${Math.round(peakDb)} dB) – näher ans Mikrofon und lauter sprechen.`);
    }
    recInfo.textContent = `Aufgenommen: ${secs} s · ${Math.round(blob.size / 1024)} KB${level} – lade hoch …`;
    submitRecording(blob, ext);
  };
  recBtn.disabled = true;
  rec.start(1000);  // in 1-s-Stücken: iOS/Safari verliert so am Ende nichts
  let left = REC_SECONDS;
  const label = () => {
    const lv = meter ? `  ·  Pegel ${Math.round(meter.read())} dB` : "";
    recBtn.textContent = `● Aufnahme … ${left} s${lv}`;
  };
  label();
  recInfo.textContent = "Jetzt sprechen – ruhig und deutlich, ganze Sätze.";
  const levelTimer = meter ? setInterval(label, 200) : null;
  const timer = setInterval(() => {
    left -= 1;
    if (left <= 0) { clearInterval(timer); if (levelTimer) clearInterval(levelTimer); rec.stop(); }
  }, 1000);
});

// Name: nur erlaubte Zeichen (klein, ohne Leerzeichen)
el.cloneName.addEventListener("input", () => {
  el.cloneName.value = el.cloneName.value.toLowerCase().replace(/[^a-z0-9_-]/g, "");
});
el.cloneTextSave.addEventListener("click", async () => {
  const res = await fetch("/api/clone/text", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: el.cloneResult.dataset.name, text: el.cloneText.value }),
  });
  if (res.ok) { hideError(); el.cloneTextSave.textContent = "Gespeichert ✓"; setTimeout(() => (el.cloneTextSave.textContent = "Text speichern"), 2000); }
  else { const b = await res.json().catch(() => ({})); showError(b.detail || "Speichern fehlgeschlagen"); }
});

/* ------------------------------------------ Ausschnitte zur Auswahl */
// Ein Player für alle Ausschnitte: ▶ startet, nochmal klicken stoppt
const candAudio = new Audio();
let candPlaying = null;
candAudio.addEventListener("ended", () => setCandPlaying(null));

function setCandPlaying(btn) {
  if (candPlaying && candPlaying !== btn) candPlaying.textContent = "▶";
  candPlaying = btn;
  if (btn) btn.textContent = "■";
}

function fmtSec(s) {
  return `${s.toFixed(1).replace(".", ",")} s`;
}

async function loadCandidates(name) {
  const data = await fetch(`/api/clone/candidates/${encodeURIComponent(name)}`)
    .then((r) => r.json()).catch(() => null);
  el.cloneCands.innerHTML = "";
  const list = (data && data.candidates) || [];
  el.cloneCandsBox.classList.toggle("hidden", list.length < 2);
  for (const c of list) {
    const li = document.createElement("li");
    const play = document.createElement("button");
    play.type = "button";
    play.textContent = "▶";
    play.title = "Anhören";
    play.addEventListener("click", () => {
      if (candPlaying === play) { candAudio.pause(); setCandPlaying(null); return; }
      candAudio.src = `/api/clone/candidates/${encodeURIComponent(name)}/${c.idx}.wav?t=${Date.now()}`;
      candAudio.play().then(() => setCandPlaying(play)).catch((e) => showError("Abspielen nicht möglich: " + e));
    });
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = `${c.idx} · ${fmtSec(c.seconds)} · ${c.text}`;
    meta.title = c.text;
    const pick = document.createElement("button");
    pick.type = "button";
    const active = data.chosen === c.idx;
    pick.textContent = active ? "✓ gewählt" : "Diesen nehmen";
    pick.disabled = active;
    pick.addEventListener("click", async () => {
      pick.disabled = true;
      const res = await fetch(`/api/clone/candidates/${encodeURIComponent(name)}/${c.idx}`, { method: "POST" });
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        showError(b.detail || "Übernehmen fehlgeschlagen");
        pick.disabled = false;
        return;
      }
      const r = await res.json();
      hideError();
      el.cloneText.value = r.text;
      loadCandidates(name);
    });
    li.append(play, meta, pick);
    el.cloneCands.append(li);
  }
}

/* ------------------------------------------------------------------ Init */
(async function init() {
  applyConfig(await fetch("/api/config").then((r) => r.json()));
  pollClone(); // läuft gerade eine Erstellung (z. B. nach Neuladen)? Dann weiter anzeigen
  fetch("/api/clone").then((r) => r.json()).then((j) => { if (j.running && !clonePoll) clonePoll = setInterval(pollClone, 1500); }).catch(() => {});
  connectWs();
  setInterval(refreshStatus, 5000);
})();

// ---------------------------------------------------- eigene Stimmen löschen
async function loadCloneList() {
  const list = await fetch("/api/clone/voices").then((r) => r.json()).catch(() => null);
  if (!list) return;
  el.cloneList.innerHTML = "";
  if (!list.length) {
    el.cloneList.innerHTML = '<li class="muted">Keine eigenen Stimmen.</li>';
    return;
  }
  for (const v of list) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = v.name;
    const meta = document.createElement("span");
    meta.className = "meta muted";
    meta.textContent = `${v.kb} KB · ${v.text}`;
    meta.title = v.text;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "Löschen";
    btn.addEventListener("click", async () => {
      if (!window.confirm(`Eigene Stimme „${v.name}“ endgültig löschen?`)) return;
      btn.disabled = true;
      const res = await fetch(`/api/clone/voices/${encodeURIComponent(v.name)}`, { method: "DELETE" });
      if (!res.ok) {
        btn.disabled = false;
        const err = await res.json().catch(() => ({}));
        showError(err.detail || "Löschen fehlgeschlagen.");
        return;
      }
      const data = await res.json();
      el.voice.dataset.selected = data.voice;
      if (el.cloneResult.dataset.name === v.name) el.cloneResult.classList.add("hidden");
      // Liste/Dropdown aktualisiert die voices_changed-Nachricht
    });
    // Ausschnitt nachträglich wechseln (falls beim Erstellen Kandidaten entstanden)
    const cands = document.createElement("button");
    cands.type = "button";
    cands.textContent = "Ausschnitt";
    cands.title = "Ausschnitte anhören und einen anderen übernehmen";
    cands.addEventListener("click", () => {
      el.cloneResult.dataset.name = v.name;
      el.cloneText.value = v.text;
      el.cloneResult.classList.remove("hidden");
      loadCandidates(v.name);
      el.cloneResult.scrollIntoView({ behavior: "smooth", block: "center" });
    });
    li.append(name, meta, cands, btn);
    el.cloneList.append(li);
  }
}
loadCloneList();

// ------------------------------------------------------ Stimmen verwalten
async function setVisibility(changes) {
  const res = await fetch("/api/voices/visibility", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(changes),
  });
  if (!res.ok) { showError("Ändern der Stimmenliste fehlgeschlagen."); return; }
  const data = await res.json();
  el.voice.dataset.selected = data.voice;
  // Dropdown + Liste aktualisiert die voices_changed-Nachricht
}

el.hideVoice.addEventListener("click", () => {
  const opt = el.voice.selectedOptions[0];
  if (!opt) return;
  if (el.voice.options.length <= 1) { showError("Die letzte Stimme lässt sich nicht entfernen."); return; }
  if (!window.confirm(`„${opt.textContent}“ aus der Liste entfernen?\n\nÜber „Stimmen verwalten …“ jederzeit wieder einblendbar.`)) return;
  setVisibility({ [opt.value]: false });
});

el.manageVoices.addEventListener("click", () => {
  const show = el.voiceManager.classList.toggle("hidden") === false;
  el.manageVoices.textContent = show ? "Stimmen verwalten schließen" : "Stimmen verwalten …";
  if (show) {
    loadVoiceCatalog();
    fetch("/api/share/state").then((r) => r.json()).then(showShareState).catch(() => null);
  }
});

const openGroups = new Set();
async function loadVoiceCatalog() {
  const groups = await fetch("/api/voices/catalog").then((r) => r.json()).catch(() => null);
  if (!groups) return;
  el.voiceGroups.innerHTML = "";
  for (const g of groups) {
    const det = document.createElement("details");
    det.open = openGroups.has(g.group);
    det.addEventListener("toggle", () => { det.open ? openGroups.add(g.group) : openGroups.delete(g.group); });
    const sum = document.createElement("summary");
    const active = g.voices.filter((v) => v.visible).length;
    sum.textContent = g.group + " ";
    const cnt = document.createElement("span");
    cnt.className = "muted";
    cnt.textContent = `(${active} von ${g.voices.length} aktiv)`;
    sum.append(cnt);
    det.append(sum);
    for (const v of g.voices) {
      const row = document.createElement("div");
      row.className = "vm-item";
      row.dataset.search = `${g.group} ${v.label}${v.shared ? " freigegeben" : ""}`.toLowerCase();
      const lab = document.createElement("label");
      lab.className = "vm-show";
      lab.title = "In der Stimmauswahl anzeigen";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = v.visible;
      cb.addEventListener("change", () => setVisibility({ [v.key]: cb.checked }));
      const txt = document.createElement("span");
      txt.textContent = v.label;
      lab.append(cb, txt);
      const shareLab = document.createElement("label");
      shareLab.className = "vm-share";
      shareLab.title = "Für die Freunde-Instanz freigeben";
      const sb = document.createElement("input");
      sb.type = "checkbox";
      sb.checked = !!v.shared;
      sb.addEventListener("change", () => setShared({ [v.key]: sb.checked }));
      shareLab.append(sb, document.createTextNode("Freigeben"));
      row.append(lab, shareLab);
      const state = v.download_state || (v.visible && v.download ? "wird geladen …" : v.download ? "wird beim Aktivieren geladen" : "");
      if (state) {
        const st = document.createElement("span");
        st.className = "vm-state" + (state.startsWith("Download fehl") ? " err" : "");
        st.textContent = state === "läuft" ? "wird geladen …" : state;
        lab.append(st);
      }
      det.append(row);
    }
    el.voiceGroups.append(det);
  }
  applyVoiceFilter();
}

async function setShared(changes) {
  const r = await fetch("/api/voices/shared", {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(changes),
  }).then((x) => x.json()).catch(() => null);
  if (r) showShareState(r);
}

function showShareState(st) {
  el.shareStatus.textContent = st.pending
    ? `${st.draft} freigegeben – Änderungen noch nicht übernommen`
    : `${st.published} Stimme${st.published === 1 ? "" : "n"} in der Produktiv-Instanz`;
  el.sharePublish.classList.toggle("btn-primary", !!st.pending);
}

function fillShareInfo(cfg) {
  if (document.activeElement !== el.prodUrl) el.prodUrl.value = cfg.prod_url || "";
  if (document.activeElement !== el.shareSelfUrl) {
    // Hinter einem Reverse-Proxy passt die Browser-Adresse nicht – nur IP-Adressen vorschlagen
    const guess = /^\d+\.\d+\.\d+\.\d+$/.test(location.hostname) ? location.origin : "";
    el.shareSelfUrl.value = cfg.share_self_url || guess;
  }
  el.shareToken.placeholder = cfg.share_token_set ? "gespeichert" : "";
}

async function saveShareSettings() {
  const changes = { prod_url: el.prodUrl.value, share_self_url: el.shareSelfUrl.value };
  if (el.shareToken.value.trim()) changes.share_token = el.shareToken.value.trim();
  const c = await fetch("/api/config", {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(changes),
  }).then((x) => x.json()).catch(() => null);
  if (c) { el.shareToken.value = ""; fillShareInfo(c); }
}
for (const input of [el.prodUrl, el.shareToken, el.shareSelfUrl]) {
  input.addEventListener("change", saveShareSettings);
}

el.sharePublish.addEventListener("click", async () => {
  el.sharePublish.disabled = true;
  el.shareStatus.textContent = "Übertrage …";
  try {
    await saveShareSettings();
    const res = await fetch("/api/share/publish", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ self_url: el.shareSelfUrl.value }),
    });
    const r = await res.json().catch(() => ({}));
    if (res.ok) showShareState(r);
    else el.shareStatus.textContent = r.detail || "Übertragung fehlgeschlagen";
  } finally {
    el.sharePublish.disabled = false;
  }
});

function applyVoiceFilter() {
  const words = el.voiceFilter.value.toLowerCase().split(/\s+/).filter(Boolean);
  for (const det of el.voiceGroups.querySelectorAll("details")) {
    let hits = 0;
    for (const lab of det.querySelectorAll(".vm-item")) {
      const ok = words.every((w) => lab.dataset.search.includes(w));
      lab.classList.toggle("hidden", !ok);
      if (ok) hits++;
    }
    det.classList.toggle("hidden", hits === 0);
    if (words.length) det.open = hits > 0;
  }
}
el.voiceFilter.addEventListener("input", applyVoiceFilter);

// ------------------------------------------------------ Rechenprozesse
const workersList = $("workers-list");
const workerRows = {};
function fillWorkers(state) {
  for (const [engine, w] of Object.entries(state)) {
    let row = workerRows[engine];
    if (!row) {
      const div = document.createElement("div");
      div.className = "workers-row";
      const name = document.createElement("span");
      name.className = "wname";
      name.textContent = w.label;
      const sel = document.createElement("select");
      for (let i = 1; i <= 6; i++) {
        const o = document.createElement("option");
        o.value = String(i);
        o.textContent = i === 1 ? "1 Prozess" : `${i} Prozesse`;
        sel.append(o);
      }
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = "Anwenden";
      const st = document.createElement("span");
      st.className = "wstate";
      const note = document.createElement("p");
      note.className = "workers-note hidden";
      sel.addEventListener("change", () => { sel.dataset.dirty = "1"; });
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        st.textContent = "starte neu …";
        const res = await fetch("/api/workers", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ engine, count: Number(sel.value) }),
        }).catch(() => null);
        btn.disabled = false;
        delete sel.dataset.dirty;
        if (!res || !res.ok) {
          const err = res ? await res.json().catch(() => ({})) : {};
          showError(err.detail || "Rechenprozesse konnten nicht umgestellt werden.");
          st.textContent = "";
          return;
        }
        fillWorkers((await res.json()).workers);
      });
      div.append(name, sel, btn, st);
      workersList.append(div, note);
      row = workerRows[engine] = { sel, st, note };
    }
    if (!row.sel.dataset.dirty) row.sel.value = String(w.count);
    row.st.textContent = w.count <= 1 ? "im Hauptprozess"
      : w.running ? `${w.running} laufen` : "starten bei Bedarf";
    row.note.textContent = w.note || "";
    row.note.classList.toggle("hidden", !w.note);
  }
}
