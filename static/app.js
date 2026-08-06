"use strict";

const $ = (id) => document.getElementById(id);
const el = {
  channel: $("channel"), voice: $("voice"), oauth: $("oauth_token"),
  volume: $("volume"), volumeVal: $("volume-val"),
  speed: $("speed"), speedVal: $("speed-val"),
  readUser: $("read_username"), readEmotes: $("read_emotes"),
  cooldown: $("cooldown_seconds"), queueLimit: $("queue_limit"),
  blocklist: $("bot_blocklist"),
  skip: $("skip"), startstop: $("startstop"), audioUnlock: $("audio-unlock"),
  applyChannel: $("apply-channel"), applyVoice: $("apply-voice"),
  dot: $("dot"), statusText: $("status-text"), now: $("now-playing"),
  errorBanner: $("error-banner"),
};

/* --------------------------------------------------------- Audio-Player */
let audioCtx = null;
let gainNode = null;
let currentSource = null;

// Die Chat-Verbindung ist serverseitig global, die Browser-Audiofreigabe
// aber pro Gerät: Sie braucht zwingend eine Nutzer-Geste auf genau diesem
// Gerät (Autoplay-Sperre). Öffnet man die Seite auf einem zweiten Gerät,
// während die Verbindung schon läuft, gäbe es ohne eigenen Button nie
// Gelegenheit, den Ton freizuschalten.
function updateUnlockButton() {
  const locked = !audioCtx || audioCtx.state !== "running";
  el.audioUnlock.classList.toggle("hidden", !locked);
}

function ensureAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    gainNode = audioCtx.createGain();
    gainNode.gain.value = parseFloat(el.volume.value);
    gainNode.connect(audioCtx.destination);
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume().then(updateUnlockButton).catch(() => {});
  }
  updateUnlockButton();
}

el.audioUnlock.addEventListener("click", ensureAudio);
updateUnlockButton();

function stopPlayback() {
  if (currentSource) {
    try { currentSource.stop(); } catch (_) { /* schon beendet */ }
    currentSource = null;
  }
}

async function playUtterance(msg) {
  ensureAudio();
  stopPlayback();

  const bin = atob(msg.audio);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

  const buffer = await audioCtx.decodeAudioData(bytes.buffer);
  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(gainNode);
  source.onended = () => {
    if (currentSource === source) {
      currentSource = null;
      sendWs({ type: "played", id: msg.id });
      el.now.textContent = "–";
    }
  };
  currentSource = source;
  updateUnlockButton();
  el.now.innerHTML = "";
  const user = document.createElement("span");
  user.className = "user";
  user.textContent = msg.username;
  el.now.append(user, document.createTextNode(": " + msg.text));
  source.start();
}

/* ------------------------------------------------------------ WebSocket */
let ws = null;
let wsRetry = 1000;

function sendWs(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => { wsRetry = 1000; refreshStatus(); };
  ws.onclose = () => {
    setTimeout(connectWs, wsRetry);
    wsRetry = Math.min(wsRetry * 2, 15000);
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    switch (msg.type) {
      case "speak":
        playUtterance(msg).catch((e) => {
          console.error("Wiedergabe fehlgeschlagen", e);
          sendWs({ type: "played", id: msg.id }); // Queue nicht blockieren
        });
        break;
      case "stop_audio":
        stopPlayback();
        el.now.textContent = "–";
        break;
      case "status":
        applyStatus(msg);
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
}

/* ------------------------------------------------------- Konfiguration */
let saveTimer = null;

function collectConfig() {
  return {
    channel: el.channel.value,
    voice: el.voice.value,
    oauth_token: el.oauth.value,
    volume: parseFloat(el.volume.value),
    speed: parseFloat(el.speed.value),
    read_username: el.readUser.checked,
    read_emotes: el.readEmotes.checked,
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
  el.oauth.value = c.oauth_token;
  el.volume.value = c.volume;
  el.speed.value = c.speed;
  el.readUser.checked = c.read_username;
  el.readEmotes.checked = c.read_emotes;
  el.cooldown.value = c.cooldown_seconds;
  el.queueLimit.value = c.queue_limit;
  el.blocklist.value = c.bot_blocklist.join("\n");
  updateSliderLabels();
}

function updateSliderLabels() {
  el.volumeVal.textContent = Math.round(parseFloat(el.volume.value) * 100) + " %";
  el.speedVal.textContent = parseFloat(el.speed.value).toFixed(2) + "\u00d7";
  if (gainNode) gainNode.gain.value = parseFloat(el.volume.value);
}

/* ---------------------------------------------------------------- Events */
for (const input of [el.channel, el.voice, el.oauth, el.readUser, el.readEmotes,
                     el.cooldown, el.queueLimit, el.blocklist]) {
  input.addEventListener("change", saveConfig);
}
el.voice.addEventListener("change", () => { el.voice.dataset.selected = el.voice.value; });
for (const slider of [el.volume, el.speed]) {
  slider.addEventListener("input", () => { updateSliderLabels(); saveConfig(); });
}

el.skip.addEventListener("click", () => fetch("/api/skip", { method: "POST" }));

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
