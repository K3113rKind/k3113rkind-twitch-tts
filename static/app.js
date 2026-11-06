"use strict";

const $ = (id) => document.getElementById(id);
const el = {
  channel: $("channel"), voice: $("voice"),
  volume: $("volume"), volumeVal: $("volume-val"),
  speed: $("speed"), speedVal: $("speed-val"), speedNote: $("speed-note"),
  readUser: $("read_username"), readEmotes: $("read_emotes"),
  usernameStyle: $("username_style"),
  readMentions: $("read_mentions"), readSmileys: $("read_smileys"),
  keepAwake: $("keep_speakers_awake"),
  keepaliveLevel: $("keepalive_level"), keepaliveVal: $("keepalive-val"),
  cooldown: $("cooldown_seconds"), queueLimit: $("queue_limit"),
  blocklist: $("bot_blocklist"),
  skip: $("skip"), startstop: $("startstop"), audioUnlock: $("audio-unlock"),
  applyChannel: $("apply-channel"), applyVoice: $("apply-voice"), renameVoice: $("rename-voice"),
  dot: $("dot"), statusText: $("status-text"), now: $("now-playing"),
  errorBanner: $("error-banner"),
  sayText: $("say-text"), sayBtn: $("say"), sayCancel: $("say-cancel"),
  cloneUrl: $("clone-url"), cloneFile: $("clone-file"), cloneName: $("clone-name"),
  cloneSeconds: $("clone-seconds"), cloneStart: $("clone-start"), cloneStatus: $("clone-status"),
  cloneList: $("clone-list"), cloneResult: $("clone-result"), cloneText: $("clone-text"), cloneTextSave: $("clone-text-save"),
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
let genTimer = null;
function showGenerating(msg) {
  stopGenerating();
  const started = Date.now();
  const render = () => {
    const s = Math.floor((Date.now() - started) / 1000);
    el.now.innerHTML = "";
    const label = document.createElement("span");
    label.className = "gen";
    label.textContent = `Erzeuge Sprache … ${s} s`;
    const user = document.createElement("span");
    user.className = "user";
    user.textContent = msg.username;
    el.now.append(label, document.createElement("br"), user,
                  document.createTextNode(": " + msg.text));
  };
  render();
  genTimer = setInterval(render, 1000);
}
function stopGenerating() {
  clearInterval(genTimer);
  genTimer = null;
}

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
      case "generating":
        showGenerating(msg);
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
        // fällt durch
      case "stop_audio":
        stopGenerating();
        speaking = false;
        updateSkipButton();
        el.now.textContent = "–";
        break;
      case "status":
        applyStatus(msg);
        break;
      case "voices_changed":
        refreshStatus();
        loadCloneList();
        break;
      case "error":
        stopGenerating();
        el.now.textContent = "–";
        showError(msg.message);
        break;
    }
  };
}

/* ---------------------------------------------------------------- Status */
// Überspringen geht auch ohne laufenden Chat, z. B. bei einem Testtext.
let running = false;
let speaking = false;
function updateSkipButton() { el.skip.disabled = !running && !speaking; }

function applyStatus(s) {
  el.dot.className = "dot " + (s.connected ? "ok" : s.running ? "warn" : "");
  el.statusText.textContent = s.connected
    ? `Verbunden mit #${s.channel}`
    : s.running ? "Verbinde …" : "Nicht verbunden";
  if (s.warming) el.statusText.textContent += " · Stimme wird aufgewärmt …";
  el.startstop.textContent = s.running ? "Vorlesen stoppen" : "Vorlesen starten";
  el.startstop.classList.toggle("running", s.running);
  running = s.running;
  updateSkipButton();
  fillVoices(s.voices || []);
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
    el.voice.append(opt);
  }
  if (voices.some((v) => v.key === selected)) el.voice.value = selected;
  updateTemperatureRow();
}

// Betonungs-Regler nur bei Qwen-Stimmen zeigen
function updateTemperatureRow() {
  const opt = el.voice.selectedOptions[0];
  el.temperatureRow.classList.toggle("hidden", !(opt && opt.dataset.qwen));
  el.styleRow.classList.toggle("hidden", !(opt && opt.dataset.styleable));
  el.styleCustom.classList.toggle("hidden", el.style.value !== "eigen");
  // Qwen: Tempo nur grob per Anweisung (Presets), Klone sprechen im Tempo der Referenz
  const qwen = !!(opt && opt.dataset.qwen), styleable = !!(opt && opt.dataset.styleable);
  el.speed.disabled = qwen && !styleable;
  el.speedNote.classList.toggle("hidden", !qwen);
  el.speedNote.textContent = !qwen ? "" : styleable
    ? "Qwen: wirkt grob als Anweisung (langsamer/schneller), nicht exakt"
    : "Eigene Stimme: Tempo kommt aus der Referenzaufnahme";
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
  el.perf.innerHTML = text;
  el.perf.append(" · " + p.voice + (p.device ? ` · ${p.device}` : ""));
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
    keepalive_level: parseFloat(el.keepaliveLevel.value),
    cooldown_seconds: parseInt(el.cooldown.value || "0", 10),
    queue_limit: parseInt(el.queueLimit.value || "1", 10),
    bot_blocklist: el.blocklist.value.split("\n").map((s) => s.trim()).filter(Boolean),
    qwen_temperature: parseFloat(el.temperature.value),
    qwen_style: el.style.value,
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
  el.keepaliveLevel.value = c.keepalive_level;
  el.cooldown.value = c.cooldown_seconds;
  el.queueLimit.value = c.queue_limit;
  el.blocklist.value = c.bot_blocklist.join("\n");
  voiceLabels = c.voice_labels || {};
  el.temperature.value = c.qwen_temperature ?? 0.9;
  el.style.value = c.qwen_style || "neutral";
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
                     el.readMentions, el.readSmileys, el.keepAwake, el.cooldown, el.queueLimit, el.blocklist]) {
  input.addEventListener("change", saveConfig);
}
el.voice.addEventListener("change", () => { el.voice.dataset.selected = el.voice.value; updateTemperatureRow(); });
el.style.addEventListener("change", () => { updateTemperatureRow(); saveConfig(); });
el.styleCustom.addEventListener("change", saveConfig);

el.channel.addEventListener("change", setMediaSession);
for (const slider of [el.volume, el.speed, el.temperature]) {
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
el.sayBtn.addEventListener("click", async () => {
  const text = el.sayText.value.trim();
  if (!text) { el.sayText.focus(); return; }
  ensureAudio(); // Nutzer-Geste: löst die Autoplay-Sperre des Browsers
  el.sayBtn.disabled = true;
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
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      showError(body.detail || res.statusText);
    } else {
      hideError();
    }
  } catch (e) {
    showError("Vorlesen fehlgeschlagen: " + e);
  } finally {
    el.sayBtn.disabled = false;
  }
});

el.sayCancel.addEventListener("click", () => fetch("/api/say/cancel", { method: "POST" }));

// Strg+Enter (bzw. Cmd+Enter) im Textfeld = Vorlesen
el.sayText.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) {
    ev.preventDefault();
    el.sayBtn.click();
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
    li.append(name, meta, btn);
    el.cloneList.append(li);
  }
}
loadCloneList();
