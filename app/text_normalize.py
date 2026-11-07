"""Text für Qwen/Pocket aufbereiten: Zahlen, Uhrzeiten, Daten, Abkürzungen,
Einheiten, Links und Emojis in ausgeschriebenes Deutsch.

An solchen Stellen verhaspeln sich die neuronalen Modelle oder verschlucken
etwas ("2026", "z. B.", "14:30", "50%"). Kokoro braucht das nicht – dessen
Phonemisierung (espeak) liest Zahlen selbst.
"""

from __future__ import annotations

import re

_UNITS = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
          "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn",
          "achtzehn", "neunzehn"]
_TENS = {2: "zwanzig", 3: "dreißig", 4: "vierzig", 5: "fünfzig", 6: "sechzig",
         7: "siebzig", 8: "achtzig", 9: "neunzig"}
_MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
           "September", "Oktober", "November", "Dezember"]


def _unit(n: int) -> str:
    """Zahl 1–9 als Teil eines Kompositums ("ein" statt "eins")."""
    return "ein" if n == 1 else _UNITS[n]


def _below100(n: int, final: bool = True) -> str:
    if n < 20:
        return ("eins" if final else "ein") if n == 1 else _UNITS[n]
    u, t = n % 10, n // 10
    return (_unit(u) + "und" if u else "") + _TENS[t]


def _below1000(n: int, final: bool = True) -> str:
    h, r = divmod(n, 100)
    out = (_unit(h) + "hundert") if h else ""
    if r:
        out += _below100(r, final)
    return out


def number_words(n: int) -> str:
    """Ganze Zahl (0 bis < 1 Billion) als deutsches Zahlwort."""
    if n == 0:
        return "null"
    if n < 0:
        return "minus " + number_words(-n)
    if n >= 10**12:
        return " ".join(_UNITS[int(d)] for d in str(n))  # Ziffer für Ziffer
    parts = []
    bil, n = divmod(n, 10**9)
    mil, n = divmod(n, 10**6)
    tsd, n = divmod(n, 1000)
    if bil:
        parts.append("eine Milliarde" if bil == 1 else f"{_below1000(bil, False)} Milliarden")
    if mil:
        parts.append("eine Million" if mil == 1 else f"{_below1000(mil, False)} Millionen")
    small = ""
    if tsd:
        small += _below1000(tsd, False) + "tausend"
    if n:
        small += _below1000(n, True)
    if small:
        parts.append(small)
    return " ".join(parts)


def year_words(n: int) -> str:
    """Jahreszahl 1100–1999 wie gesprochen: neunzehnhundertneunundneunzig."""
    if 1100 <= n <= 1999:
        h, r = divmod(n, 100)
        return _below100(h) + "hundert" + (_below100(r) if r else "")
    return number_words(n)


def ordinal_words(n: int) -> str:
    """Ordnungszahl in der Form nach "am/vom/zum/bis zum": ersten, dritten,
    siebten, zwanzigsten – die häufigste Lesart bei Daten."""
    special = {1: "ersten", 3: "dritten", 7: "siebten", 8: "achten"}
    if n in special:
        return special[n]
    if n < 20:
        return _UNITS[n] + "ten"
    return number_words(n) + "sten"


def _decimal(m: re.Match) -> str:
    whole, frac = m.group(1), m.group(2)
    return number_words(int(whole.replace(".", ""))) + " Komma " + " ".join(_UNITS[int(d)] for d in frac)


ABBREVIATIONS = {
    r"\bz\.\s?B\.": "zum Beispiel", r"\bd\.\s?h\.": "das heißt", r"\bu\.\s?a\.": "unter anderem",
    r"\bu\.\s?U\.": "unter Umständen", r"\bo\.\s?Ä\.": "oder Ähnliches", r"\bggf\.": "gegebenenfalls",
    r"\bevtl\.": "eventuell", r"\bbzw\.": "beziehungsweise", r"\busw\.": "und so weiter",
    r"\bca\.": "circa", r"\binkl\.": "inklusive", r"\bexkl\.": "exklusive", r"\bvgl\.": "vergleiche",
    r"\bNr\.": "Nummer", r"\bDr\.": "Doktor", r"\bProf\.": "Professor", r"\bStr\.": "Straße",
    r"\bmind\.": "mindestens", r"\bmax\.": "maximal", r"\bmin\.": "minimal", r"\betc\.": "et cetera",
    r"\bsog\.": "sogenannte", r"\bbzgl\.": "bezüglich", r"\bbspw\.": "beispielsweise",
}
_ABBR = [(re.compile(k), v) for k, v in ABBREVIATIONS.items()]

_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
                    "\U0000FE0F\U0000200D\U00002B00-\U00002BFF]+")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)


_ORD_DATIVE = {"am", "vom", "zum", "zur", "im", "beim", "dem", "den", "ab", "bis", "seit"}
_ORD_NOMINATIVE = {"der", "die", "das"}
_ORD = re.compile(r"(?<![\d.,])(\d{1,3})\.(?=\s+(\S+))")


def _ordinals(t: str) -> str:
    """Ordnungszahlen: "am 7. November" -> am siebten November, "der 3. Platz"
    -> der dritte Platz. Nur nach passendem Artikel/Präposition oder vor einem
    Monatsnamen – "seit Mitte 2025. Danach" bleibt Jahreszahl mit Satzende."""
    def rep(m: re.Match) -> str:
        n = int(m.group(1))
        if n == 0:
            return m.group(0)
        before = t[:m.start()].split()
        prev = before[-1].lower() if before else ""
        if prev in _ORD_DATIVE or m.group(2).rstrip(".,!?") in _MONTHS:
            return ordinal_words(n)          # siebten
        if prev in _ORD_NOMINATIVE:
            return ordinal_words(n)[:-1]     # dritte
        return m.group(0)
    return _ORD.sub(rep, t)


def normalize(text: str) -> str:
    t = _URL.sub(" Link ", text)
    t = _EMOJI.sub(" ", t)
    for rx, repl in _ABBR:
        t = rx.sub(repl, t)
    # Datum 07.11.2026 / 7.11. -> siebter November (zweitausendsechsundzwanzig)
    def date(m: re.Match) -> str:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m.group(0)
        out = f"{ordinal_words(d)} {_MONTHS[mo - 1]}"
        if y:
            yy = int(y)
            out += " " + year_words(yy + 2000 if len(y) == 2 else yy)
        return out
    t = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})?(?!\d)", date, t)
    t = _ordinals(t)
    # Uhrzeit 14:30 (Uhr) -> vierzehn Uhr dreißig
    def clock(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        if h > 24 or mi > 59:
            return m.group(0)
        return f"{number_words(h) if h != 1 else 'ein'} Uhr" + (f" {number_words(mi)}" if mi else "")
    t = re.sub(r"\b(\d{1,2}):(\d{2})(?:\s*Uhr)?\b", clock, t)
    # Einheiten / Zeichen direkt hinter Zahlen
    # Geldbeträge 3,50 € -> drei Euro fünfzig
    def money(m: re.Match) -> str:
        euro, cent = int(m.group(1).replace(".", "")), int(m.group(2))
        return f"{number_words(euro)} Euro" + (f" {number_words(cent)}" if cent else "")
    t = re.sub(r"(\d{1,3}(?:\.\d{3})+|\d+),(\d{2})\s*€", money, t)
    t = re.sub(r"€\s*(\d{1,3}(?:\.\d{3})+|\d+),(\d{2})\b", money, t)
    t = re.sub(r"(\d)\s*%", r"\1 Prozent", t)
    t = re.sub(r"(\d)\s*€", r"\1 Euro", t)
    t = re.sub(r"€\s*(\d[\d.,]*)", r"\1 Euro", t)
    t = re.sub(r"\$\s*(\d[\d.,]*)", r"\1 Dollar", t)
    t = re.sub(r"(\d[\d.,]*)\s*\$", r"\1 Dollar", t)
    t = re.sub(r"(\d)\s*°C?\b", r"\1 Grad", t)
    t = re.sub(r"(\d)\s*km/h\b", r"\1 Kilometer pro Stunde", t)
    t = re.sub(r"(\d)\s*km\b", r"\1 Kilometer", t)
    t = re.sub(r"(\d)\s*kg\b", r"\1 Kilo", t)
    t = t.replace("&", " und ")
    # Dezimalzahl 3,5 / 1.000,25
    t = re.sub(r"\b(\d{1,3}(?:\.\d{3})+|\d+),(\d+)\b", _decimal, t)
    # Tausenderpunkte 1.000.000
    t = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b", lambda m: number_words(int(m.group(0).replace(".", ""))), t)
    # Versionsnummern u. Ä. 3.5 / 2.1.4 -> drei Punkt fünf
    t = re.sub(r"\b\d+(?:\.\d+)+\b",
               lambda m: " Punkt ".join(number_words(int(x)) for x in m.group(0).split(".")), t)
    # vierstellige Zahl allein: Jahreszahl-Lesart für 1100–1999
    t = re.sub(r"\b(1[1-9]\d\d)\b", lambda m: year_words(int(m.group(1))), t)
    # übrige ganze Zahlen (nicht innerhalb von Wörtern wie "rhd_driver2")
    t = re.sub(r"(?<![\w])(\d+)(?![\w])", lambda m: number_words(int(m.group(1))), t)
    # wiederholte Satzzeichen
    t = re.sub(r"([!?.])\1+", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
