"""deno für yt-dlp holen (Build-Schritt). Mehrere Quellen, mehrere Versuche;
klappt nichts, bricht der Build NICHT ab – dann fehlt nur deno (YouTube-Links
beim Stimmenklonen gehen evtl. nicht, Datei-Upload und alles andere schon)."""

import io
import os
import time
import urllib.request
import zipfile

ASSET = "deno-x86_64-unknown-linux-gnu.zip"


def sources():
    yield f"https://github.com/denoland/deno/releases/latest/download/{ASSET}"
    try:  # offizieller Download-Server von Deno, unabhängig von GitHub
        ver = urllib.request.urlopen("https://dl.deno.land/release-latest.txt", timeout=30).read().decode().strip()
        yield f"https://dl.deno.land/release/{ver}/{ASSET}"
    except Exception as exc:
        print(f"dl.deno.land: Version nicht ermittelbar ({exc})", flush=True)


for attempt in range(3):
    for url in sources():
        try:
            data = urllib.request.urlopen(url, timeout=120).read()
            zipfile.ZipFile(io.BytesIO(data)).extract("deno", "/usr/local/bin")
            os.chmod("/usr/local/bin/deno", 0o755)
            print(f"deno geladen von {url}", flush=True)
            raise SystemExit(0)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"deno-Download fehlgeschlagen ({url}: {exc})", flush=True)
    time.sleep(15 * (attempt + 1))

print("WARNUNG: deno nicht ladbar – Build läuft ohne weiter (YouTube-Links "
      "beim Stimmenklonen evtl. eingeschränkt). Später einfach neu bauen.", flush=True)
