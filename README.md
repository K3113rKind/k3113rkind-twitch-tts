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
   Der Tab darf im Hintergrund liegen (auch minimiert), nur nicht
   geschlossen werden.

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

- **Lautstärke** und **Sprechgeschwindigkeit**
- **Namen vorlesen** – „Peter: hallo" statt nur „hallo"
- **Emotes mitvorlesen** – normalerweise aus, sonst wird jedes Emote vorgelesen
- **Smileys und Emojis mitvorlesen** – aus entfernt Text-Smileys wie `:)`
  oder `xD` und Emojis aus der Nachricht
- **Nachrichten mit @Erwähnung vorlesen** – aus bedeutet: Nachrichten, in
  denen jemand mit `@name` angesprochen wird, werden komplett übersprungen
- **Pause je Zuschauer** – wie lange jemand warten muss, bis er wieder
  vorgelesen wird (verhindert Dauerfeuer)
- **Max. wartende Nachrichten** – bei mehr fliegen die ältesten raus,
  damit der Vorleser nicht hinterherhinkt
- **Bots, die ignoriert werden**
- **Twitch-Token** – wird nicht gebraucht, das Feld kann leer bleiben

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
- Am Handy: Stummschalter am Gerät prüfen. Handy-Browser stoppen Tabs im
  Hintergrund allerdings oft komplett – für Dauerbetrieb einen richtigen
  Rechner nehmen.

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
lokal auf dem eigenen Rechner – es geht nichts an fremde Server. Nur die
Verbindung zum Twitch-Chat geht ins Internet (anonym, ohne Anmeldung).
