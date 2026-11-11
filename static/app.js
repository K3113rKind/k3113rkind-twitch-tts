"use strict";

const $ = (id) => document.getElementById(id);
const el = {
  channel: $("channel"), voice: $("voice"),
  volume: $("volume"), volumeVal: $("volume-val"),
  speed: $("speed"), speedVal: $("speed-val"),
  readUser: $("read_username"), readEmotes: $("read_emotes"),
  usernameStyle: $("username_style"),
  readMentions: $("read_mentions"), readSmileys: $("read_smileys"),
  keepAwake: $("keep_speakers_awake"),
  poolChat: $("voice_pool_chat"), pool: $("voice-pool"), pause: $("pause"),
  keepaliveLevel: $("keepalive_level"), keepaliveVal: $("keepalive-val"),
  cooldown: $("cooldown_seconds"), queueLimit: $("queue_limit"),
  blocklist: $("bot_blocklist"),
  skip: $("skip"), startstop: $("startstop"), audioUnlock: $("audio-unlock"),
  applyChannel: $("apply-channel"), applyVoice: $("apply-voice"),
  dot: $("dot"), statusText: $("status-text"), now: $("now-playing"),
  errorBanner: $("error-banner"),
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

function showNow(msg) {
  el.now.innerHTML = "";
  const user = document.createElement("span");
  user.className = "user";
  user.textContent = msg.username;
  el.now.append(user, document.createTextNode(": " + msg.text));
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
    switch (msg.type) {
      case "speak":
        showNow(msg);
        break;
      case "done":
      case "stop_audio":
        el.now.textContent = "–";
        break;
      case "status":
        applyStatus(msg);
        break;
      case "paused":
        setPaused(msg.paused);
        break;
      case "error":
        showError(msg.message);
        break;
    }
  };
}

/* ---------------------------------------------------------------- Status */
function applyStatus(s) {
  el.dot.className = "dot " + (s.connected ? "ok" : s.running ? "warn" : "");
  el.statusText.textContent = s.connected
    ? `Verbunden mit #${s.channel}`
    : s.running ? "Verbinde …" : "Nicht verbunden";
  el.startstop.textContent = s.running ? "Vorlesen stoppen" : "Vorlesen starten";
  el.startstop.classList.toggle("running", s.running);
  el.skip.disabled = !s.running;
  if (typeof s.paused === "boolean") setPaused(s.paused);
  fillVoices(s.voices || []);
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
    el.voice.append(opt);
  }
  if (voices.some((v) => v.key === selected)) el.voice.value = selected;
  renderVoicePool(voices);
}

// Stimmen-Wechsel: Auswahl aus den lokalen Stimmen
let poolSelected = [];
let poolKeys = "";
function renderVoicePool(voices) {
  // Aufklappliste mit Häkchen (mehrere Stimmen wählbar)
  if (!el.pool) return;
  const list = voices;
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

// Pause-Knopf
let paused = false;
function setPaused(p) {
  paused = !!p;
  el.pause.textContent = paused ? "▶ Weiter" : "⏸ Pause";
  el.pause.classList.toggle("running", paused);
}

/* ------------------------------------------------------- Konfiguration */
let saveTimer = null;

function collectConfig() {
  return {
    channel: el.channel.value,
    voice: el.voice.value,
    volume: parseFloat(el.volume.value),
    speed: parseFloat(el.speed.value),
    read_username: el.readUser.checked,
    username_style: el.usernameStyle.value,
    read_emotes: el.readEmotes.checked,
    read_mentions: el.readMentions.checked,
    read_smileys: el.readSmileys.checked,
    keep_speakers_awake: el.keepAwake.checked,
    voice_pool_chat: el.poolChat.checked,
    voice_pool: poolSelected,
    keepalive_level: parseFloat(el.keepaliveLevel.value),
    cooldown_seconds: parseInt(el.cooldown.value || "0", 10),
    queue_limit: parseInt(el.queueLimit.value || "1", 10),
    bot_blocklist: el.blocklist.value.split("\n").map((s) => s.trim()).filter(Boolean),
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
  el.poolChat.checked = !!c.voice_pool_chat;
  poolSelected = Array.isArray(c.voice_pool) ? c.voice_pool : [];
  if (el.pool) el.pool.querySelectorAll("input").forEach((cb) => { cb.checked = poolSelected.includes(cb.value); });
  updatePoolButton();
  el.keepaliveLevel.value = c.keepalive_level;
  el.cooldown.value = c.cooldown_seconds;
  el.queueLimit.value = c.queue_limit;
  el.blocklist.value = c.bot_blocklist.join("\n");
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
}

/* ---------------------------------------------------------------- Events */
for (const input of [el.channel, el.voice, el.readUser, el.usernameStyle, el.readEmotes,
                     el.readMentions, el.readSmileys, el.keepAwake, el.poolChat, el.cooldown, el.queueLimit, el.blocklist]) {
  input.addEventListener("change", saveConfig);
}
el.voice.addEventListener("change", () => { el.voice.dataset.selected = el.voice.value; });
el.channel.addEventListener("change", setMediaSession);
for (const slider of [el.volume, el.speed]) {
  slider.addEventListener("input", () => { updateSliderLabels(); saveConfig(); });
}

el.skip.addEventListener("click", () => fetch("/api/skip", { method: "POST" }));
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
const previewBtn = document.getElementById("preview-voice");
previewBtn.addEventListener("click", () => applyNow(previewBtn, "/api/preview"));

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

/* ------------------------------------------------------------------ Init */
(async function init() {
  applyConfig(await fetch("/api/config").then((r) => r.json()));
  connectWs();
  setInterval(refreshStatus, 5000);
})();
