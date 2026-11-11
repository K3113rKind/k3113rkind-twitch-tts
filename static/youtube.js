/* YouTube neu vertonen: Gerechnet wird auf der Administrativ-Instanz, hier
   wird nur der Auftrag gestellt und das fertige Video angezeigt. Ein Video zur
   Zeit für alle; solange eins bearbeitet wird, ist "Laden & vertonen" gesperrt.
   Nutzt aus app.js: ensureAudio(), showError(), hideError(). */
(() => {
  const $ = (id) => document.getElementById(id);
  const ui = {
    url: $("yt-url"), orig: $("yt-orig"), height: $("yt-height"), start: $("yt-start"),
    end: $("yt-end"), load: $("yt-load"), info: $("yt-info"), wrap: $("yt-wrap"),
    video: $("yt-video"), eject: $("yt-eject"),
  };
  if (!ui.load || !ui.video) return;
  let poll = null;
  let shownId = null;

  // Kein Speichern über das Kontextmenü (verhindert nicht alles, nur das Offensichtliche)
  ui.video.addEventListener("contextmenu", (ev) => ev.preventDefault());

  const hms = (s) => {
    s = Math.max(0, Math.round(s || 0));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
  };

  function showVideo(meta, expiresIn) {
    shownId = meta.id;
    ui.info.classList.remove("hidden");
    ui.info.textContent = `${meta.title}` + (meta.channel ? ` – ${meta.channel}` : "") +
      ` · ${hms(meta.duration)} min · Original: ${meta.language} · ${meta.sentences} Sätze` +
      (expiresIn != null ? ` · wird in ${Math.floor(expiresIn / 3600)} h ${Math.floor((expiresIn % 3600) / 60)} min gelöscht` : "");
    ui.video.src = `/api/youtube/video?id=${encodeURIComponent(meta.id || "")}&t=${Date.now()}`;
    ui.wrap.classList.remove("hidden");
    ui.eject.classList.remove("hidden");
  }

  function hideVideo() {
    shownId = null;
    ui.video.pause();
    ui.video.removeAttribute("src");
    ui.video.load();
    ui.wrap.classList.add("hidden");
    ui.eject.classList.add("hidden");
    ui.info.classList.add("hidden");
    ui.info.textContent = "";
  }

  function schedule(ms) {
    clearTimeout(poll);
    poll = setTimeout(pollJob, ms);
  }

  async function pollJob() {
    const res = await fetch("/api/youtube/job").catch(() => null);
    if (!res || !res.ok) {
      const body = res ? await res.json().catch(() => ({})) : {};
      ui.load.disabled = true;
      ui.load.title = body.detail || "Administrativ-Instanz nicht erreichbar";
      ui.info.classList.remove("hidden");
      ui.info.textContent = body.detail || "YouTube ist gerade nicht erreichbar.";
      schedule(30000);
      return;
    }
    ui.load.title = "";
    const job = await res.json();
    if (job.running) {
      ui.load.disabled = true;
      if (shownId) hideVideo();
      ui.eject.classList.remove("hidden");
      ui.info.classList.remove("hidden");
      ui.info.textContent = "In Bearbeitung: " + (job.log && job.log.length ? job.log[job.log.length - 1] : "Starte …") +
        (job.url ? ` · ${job.url}` : "");
      schedule(2000);
      return;
    }
    ui.load.disabled = false;
    if (job.error) {
      if (shownId) hideVideo();
      ui.info.classList.remove("hidden");
      ui.info.textContent = "Fehler: " + job.error;
      ui.eject.classList.add("hidden");
    } else if (job.meta) {
      if (shownId !== job.meta.id) showVideo(job.meta, job.meta.finished ? job.meta.finished + 12 * 3600 - Date.now() / 1000 : null);
    } else {
      hideVideo();
    }
    schedule(15000);  // auch andere Nutzer sehen so, wenn jemand etwas startet
  }

  ui.load.addEventListener("click", async () => {
    const url = ui.url.value.trim();
    if (!url) { ui.url.focus(); return; }
    ui.load.disabled = true;
    const res = await fetch("/api/youtube/prepare", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, original_level: parseFloat(ui.orig.value), height: parseInt(ui.height.value, 10),
                             start: ui.start.value.trim(), end: ui.end.value.trim() }),
    }).catch(() => null);
    if (!res || !res.ok) {
      const body = res ? await res.json().catch(() => ({})) : {};
      showError(body.detail || "Laden fehlgeschlagen.");
      ui.load.disabled = false;
      return;
    }
    hideError();
    hideVideo();
    pollJob();
  });

  ui.eject.addEventListener("click", async () => {
    if (!window.confirm("Video abbrechen bzw. vom Server löschen?")) return;
    hideVideo();
    await fetch("/api/youtube/eject", { method: "POST" }).catch(() => {});
    pollJob();
  });

  pollJob();
})();
