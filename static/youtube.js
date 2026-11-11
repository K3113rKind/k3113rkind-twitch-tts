/* YouTube vorlesen: Der Server lädt das Video (höchstens 720p), besorgt den
   Text (deutsche Untertitel, sonst Originalsprache bzw. Whisper), übersetzt
   lokal, vertont alle Sätze mit der eingestellten Stimme und setzt die
   deutsche Spur ins Video ein (Original bleibt als 2. Spur). Erst dann wird
   das Video hier angeboten – frei spulbar, als Datei exportierbar.
   "Auswerfen" bricht ab bzw. löscht alles vom Server; ein neues Video
   ersetzt das alte. Nutzt aus app.js: ensureAudio(), showError(), hideError(). */
(() => {
  const $ = (id) => document.getElementById(id);
  const ui = {
    url: $("yt-url"), orig: $("yt-orig"), height: $("yt-height"),
    start: $("yt-start"), end: $("yt-end"), load: $("yt-load"), info: $("yt-info"),
    wrap: $("yt-wrap"), video: $("yt-video"), exp: $("yt-export"), eject: $("yt-eject"),
  };
  if (!ui.load || !ui.video) return;
  let poll = null;
  let shownId = null;

  const post = (url, body) => fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}),
  });

  function fmtDur(s) {
    if (!s) return "";
    const m = Math.floor(s / 60), sec = Math.round(s % 60);
    return ` · ${m}:${String(sec).padStart(2, "0")} min`;
  }

  function showVideo(meta) {
    shownId = meta.id;
    ui.info.classList.remove("hidden");
    ui.info.textContent = `${meta.title}` + (meta.channel ? ` – ${meta.channel}` : "") +
      fmtDur(meta.duration) + ` · Original: ${meta.language} · ${meta.source} · ` +
      `${(meta.sentences || []).length} Sätze` +
      (meta.sped_up ? ` · ${meta.sped_up} beschleunigt` : "") +
      (meta.max_delay ? ` · max. ${meta.max_delay} s Versatz` : "");
    ui.video.src = `/api/youtube/video?id=${encodeURIComponent(meta.id || "")}&t=${Date.now()}`;
    ui.wrap.classList.remove("hidden");
    ui.exp.classList.remove("hidden");
    ui.eject.classList.remove("hidden");
  }

  function hideVideo() {
    shownId = null;
    ui.video.pause();
    ui.video.removeAttribute("src");
    ui.video.load();
    ui.wrap.classList.add("hidden");
    ui.exp.classList.add("hidden");
    ui.eject.classList.add("hidden");
    ui.info.classList.add("hidden");
    ui.info.textContent = "";
  }

  async function pollJob() {
    const res = await fetch("/api/youtube/job").catch(() => null);
    if (!res || !res.ok) return;
    const job = await res.json();
    if (job.url && !ui.url.value) ui.url.value = job.url;  // nach Tab neu öffnen
    if (job.running) {
      ui.load.disabled = true;
      ui.eject.classList.remove("hidden");
      ui.info.classList.remove("hidden");
      ui.info.textContent = job.log && job.log.length ? job.log[job.log.length - 1] : "Starte …";
      if (!poll) poll = setInterval(pollJob, 1500);
      return;
    }
    clearInterval(poll); poll = null;
    ui.load.disabled = false;
    if (job.error) {
      hideVideo();
      showError(job.error);
    } else if (job.meta && shownId !== job.meta.id) {
      hideError();
      showVideo(job.meta);
    } else if (!job.meta && shownId === null) {
      ui.eject.classList.add("hidden");
      ui.info.classList.add("hidden");
    }
  }

  ui.load.addEventListener("click", async () => {
    const url = ui.url.value.trim();
    if (!url) { ui.url.focus(); return; }
    hideVideo();
    ui.load.disabled = true;
    const res = await post("/api/youtube/prepare",
                           { url, original_level: parseFloat(ui.orig.value),
                             height: parseInt(ui.height.value, 10),
                             start: ui.start.value.trim(), end: ui.end.value.trim() }).catch(() => null);
    if (!res || !res.ok) {
      const body = res ? await res.json().catch(() => ({})) : {};
      showError(body.detail || "Laden fehlgeschlagen.");
      ui.load.disabled = false;
      return;
    }
    hideError();
    pollJob();
  });

  ui.eject.addEventListener("click", async () => {
    hideVideo();
    await post("/api/youtube/eject").catch(() => {});
    clearInterval(poll); poll = null;
    ui.load.disabled = false;
  });

  // Not-Aus aus app.js: Anzeige zurücksetzen
  window.addEventListener("tts-emergency", () => {
    hideVideo();
    clearInterval(poll); poll = null;
    ui.load.disabled = false;
  });

  // Auflösung merken (nur in diesem Browser)
  try {
    const h = localStorage.getItem("yt-height");
    if (h) ui.height.value = h;
  } catch (e) { /* ohne Speicher eben Standard */ }
  ui.height.addEventListener("change", () => {
    try { localStorage.setItem("yt-height", ui.height.value); } catch (e) { /* egal */ }
  });

  // Beim Öffnen der Seite: läuft eine Vorbereitung oder liegt schon ein Video bereit?
  pollJob();
})();

/* Video der Freunde-Instanz: Stand anzeigen, ansehen, exportieren, löschen.
   Wird dort nach 12 h automatisch gelöscht. */
(() => {
  const $ = (id) => document.getElementById(id);
  const box = $("ytf-box"), info = $("ytf-info"), wrap = $("ytf-wrap"), video = $("ytf-video");
  const show = $("ytf-show"), exp = $("ytf-export"), eject = $("ytf-eject");
  if (!box) return;
  let shownId = null, timer = null;

  const hms = (s) => {
    s = Math.max(0, Math.round(s || 0));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
  };

  function hideVideo() {
    video.pause();
    video.removeAttribute("src");
    video.load();
    wrap.classList.add("hidden");
    shownId = null;
  }

  async function poll() {
    const job = await fetch("/api/youtube/job?slot=freunde").then((r) => r.json()).catch(() => null);
    clearTimeout(timer);
    timer = setTimeout(poll, job && job.running ? 3000 : 15000);
    if (!job) return;
    const m = job.meta;
    if (!job.running && !m && !job.error) {
      box.classList.add("hidden");
      hideVideo();
      return;
    }
    box.classList.remove("hidden");
    if (job.running) {
      info.textContent = "In Bearbeitung: " + (job.log && job.log.length ? job.log[job.log.length - 1] : "Starte …") +
        (job.url ? ` · ${job.url}` : "");
    } else if (job.error) {
      info.textContent = "Fehler: " + job.error;
    } else {
      const left = job.expires_in != null ? ` · wird in ${Math.floor(job.expires_in / 3600)} h ${Math.floor((job.expires_in % 3600) / 60)} min gelöscht` : "";
      info.textContent = `${m.title}` + (m.channel ? ` – ${m.channel}` : "") + ` · ${hms(m.duration)} min` + left;
      if (shownId && shownId !== m.id) hideVideo();
      box.dataset.id = m.id || "";
    }
    const ready = !job.running && !!m;
    show.classList.toggle("hidden", !ready);
    exp.classList.toggle("hidden", !ready);
    if (!ready) hideVideo();
  }

  show.addEventListener("click", () => {
    if (!wrap.classList.contains("hidden")) { hideVideo(); return; }
    video.src = `/api/youtube/video?slot=freunde&t=${Date.now()}`;
    wrap.classList.remove("hidden");
    shownId = box.dataset.id || "";
  });
  eject.addEventListener("click", async () => {
    if (!window.confirm("Video der Freunde-Instanz abbrechen bzw. vom Server löschen?")) return;
    hideVideo();
    await fetch("/api/youtube/eject?slot=freunde", { method: "POST" }).catch(() => {});
    poll();
  });
  poll();
})();
