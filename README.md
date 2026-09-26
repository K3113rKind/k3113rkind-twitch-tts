# K3113rkind's Twitch TTS

Liest den Twitch-Chat vor. Vier Stimmen zur Auswahl, sonst nichts
einzustellen – keine Modelle aussuchen, nichts extra herunterladen.

## Installation

> **Hinweis:** Läuft aktuell nur unter Linux. Eine Windows-Variante ist
> geplant, aber noch nicht fertig.

Ordner entpacken, Terminal darin öffnen, eintippen:

```bash
./install.sh
```

Das war's. Das Skript kümmert sich um alles: Docker (falls nötig),
Stimmen herunterladen, Programm starten, Desktop-Symbol anlegen.
Beim ersten Mal dauert es einige Minuten.

Danach im Browser öffnen: **http://localhost:8380**

## Benutzen

1. Twitch-Kanal eintragen (nur den Namen, z. B. `bonjwa`)
2. Stimme auswählen
3. Auf **Vorlesen starten** klicken
4. Den Browser-Tab geöffnet lassen – der Ton kommt aus diesem Fenster.
   Der Tab darf im Hintergrund liegen (auch minimiert), am Handy darf auch
   das Display gesperrt sein – nur nicht schließen.

Der Ton kommt als durchgehender Audio-Stream vom Server (wie ein
Internetradio, Adresse `/stream`). Dadurch spielt auch das iPhone im
Hintergrund weiter, und auf dem Sperrbildschirm lässt sich die Wiedergabe
pausieren. Der Ton läuft dabei etwa 1–2 Sekunden hinter der Textanzeige.

Alles wird automatisch gespeichert.

## Stimmen

| Stimme   | Sprache  |         |
| -------- | -------- | ------- |
| Victoria | Deutsch  | weiblich |
| Martin   | Deutsch  | männlich |
| Heart    | Englisch | weiblich |
| Michael  | Englisch | männlich |

Die deutschen und die englischen Stimmen sind verschiedene Modelle –
eine deutsche Stimme kann kein Englisch und umgekehrt. Bei gemischten
Chats einfach die Stimme nehmen, die zur Hauptsprache passt.

## Weitere Einstellungen

Aufklappbar unter „Weitere Einstellungen":

- **Lautstärke** und **Sprechgeschwindigkeit** (die Lautstärke gilt für alle
  verbundenen Geräte, weil sie im Stream eingerechnet wird)
- **Namen vorlesen** – „Peter: hallo" statt nur „hallo"
- **Wie der Name angekündigt wird** – „Peter: hallo" oder „Peter sagt hallo"
  (bei englischen Stimmen „Peter says hallo")
- **Twitch-Emotes mitvorlesen** – betrifft Emotes wie `Kappa`, `PogChamp`
  oder `KEKW`; normalerweise aus, sonst wird jedes einzelne vorgelesen.
  Erfasst auch Emotes von BetterTTV, 7TV und FrankerFaceZ: deren Namenslisten
  werden beim ersten Betreten eines Kanals automatisch geladen. Die ersten
  ein, zwei Nachrichten danach können noch durchrutschen.
- **Smileys und Emojis mitvorlesen** – betrifft getippte Zeichen wie `:)`,
  `xD` und Unicode-Emojis wie 😀; aus entfernt sie aus der Nachricht
- **Hintergrundton** – ein durchgehender, sehr leiser Ton im Stream. Er sorgt
  dafür, dass der Browser den Tab als tonausgebend einstuft. Richtig
  eingestellt ist er, wenn das Lautsprechersymbol am Tab dauerhaft leuchtet
  und man nichts hört. 0 schaltet ihn ab
- **Lautsprecher wach halten** – sendet nach 45 Sekunden Stille einen
  kurzen, sehr leisen Tiefton. Verhindert, dass Bluetooth-Boxen oder Soundbars bei
  längerer Stille abschalten und den Anfang der nächsten Ansage
  verschlucken. Bei Kopfhörern nicht nötig (die schlafen nicht ein) – wer
  den Impuls hört, schaltet die Option einfach aus. Feineinstellung über
  die Konstanten `WAKE_*` in `app/stream.py`
- **Nachrichten mit @Erwähnung vorlesen** – aus bedeutet: Nachrichten, in
  denen jemand mit `@name` angesprochen wird, werden komplett übersprungen
- **Pause je Zuschauer** – wie lange jemand warten muss, bis er wieder
  vorgelesen wird (verhindert Dauerfeuer)
- **Max. wartende Nachrichten** – bei mehr fliegen die ältesten raus,
  damit der Vorleser nicht hinterherhinkt
- **Bots, die ignoriert werden**

## In OBS einbinden

In OBS eine **Browserquelle** hinzufügen und als URL eintragen:

```
http://localhost:8380/overlay
```

Breite/Höhe nach Geschmack (z. B. 1920 × 200). Wichtig: in den Eigenschaften
der Quelle **"Audio über OBS steuern"** aktivieren, damit der Ton im Stream
landet und nicht nur lokal läuft.

Die Seite zeigt die gerade vorgelesene Nachricht als Einblendung. Wer nur
den Ton will, hängt `?text=0` an die URL:

```
http://localhost:8380/overlay?text=0
```

Bedient wird weiterhin über die normale Seite (`http://localhost:8380`) –
das Overlay hat bewusst keine Bedienelemente.

Falls kein Ton kommt: In OBS Rechtsklick auf die Quelle → **Interagieren**
→ einmal ins Bild klicken. Manche OBS-Versionen verlangen diesen einen
Klick, bevor sie Ton abspielen.

## Kein Ton?

- Auf **Ton auf diesem Gerät einschalten** klicken, falls der Knopf da ist
- Browser-Tab muss geöffnet bleiben (im Hintergrund ist in Ordnung)
- Am iPhone: Nach dem Öffnen der Seite einmal **Ton auf diesem Gerät
  einschalten** oder **Vorlesen starten** antippen – iOS verlangt diesen
  Tipp, bevor Ton kommen darf. Danach darf das Display gesperrt werden.
- Läuft die Seite über einen Reverse-Proxy (z. B. Pangolin/Traefik), darf
  der `/stream` nicht gepuffert oder komprimiert werden.

## Wieder starten / beenden

Desktop-Symbol **K3113rkind's Twitch TTS** anklicken. Zum Beenden das Fenster
schließen oder Strg+C drücken.

Ohne Desktop-Symbol geht es auch im Terminal im Projektordner:

```bash
./start.sh
```

## Lizenz

Freie Software unter der **GPL-3.0-or-later**. Forks und Weitergabe sind
ausdrücklich erwünscht, solange der Copyright-Vermerk (Emanuel Höft /
K3113rKind) erhalten bleibt, Änderungen gekennzeichnet werden und die
weitergegebene Version ebenfalls unter der GPL steht.

Details und die Gründe für die Lizenzwahl stehen in der Datei `LICENSE`.

## Was steckt drin?

[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) als Sprachmodell,
die deutschen Stimmen kommen aus dem
[kikiri-tts-Projekt](https://github.com/semidark/kikiri-tts). Alles läuft
lokal auf dem eigenen Rechner, die Sprachausgabe verlässt ihn nicht. Ins
Internet gehen nur die Verbindung zum Twitch-Chat (anonym, ohne Anmeldung)
und der einmalige Abruf der öffentlichen Emote-Namenslisten von BetterTTV,
7TV und FrankerFaceZ je Kanal (abschaltbar, indem „Twitch-Emotes
mitvorlesen" aktiviert wird).

Die Oberfläche ist ohne Anmeldung im ganzen Heimnetz erreichbar
(Port 8380) – gewollt, damit sie auch vom Handy oder Tablet bedienbar ist.
Nur für diesen Rechner: in `docker-compose.yml` die Zeile
`"8380:8000"` in `"127.0.0.1:8380:8000"` ändern.
