"""Satzteilung und Textaufbereitung."""
from app.tts import split_for_qwen
from app.text_normalize import normalize


def test_never_splits_inside_sentence():
    text = ("Das ist der erste Satz. Und hier kommt ein zweiter, etwas längerer Satz mit Kommas, "
            "der weitergeht. Kurz. Noch ein Satz zum Schluss!")
    chunks = split_for_qwen(text, first_max=60)
    assert " ".join(chunks).split() == text.split()
    for c in chunks:
        assert c.rstrip()[-1] in ".!?", c


def test_number_dot_is_no_sentence_end():
    chunks = split_for_qwen("Am 7. November war ich auf Platz 3. Danach ging es weiter.", first_max=20)
    assert not any(c.endswith("7.") for c in chunks)
    assert any(c.startswith("Am 7. November") for c in chunks)


def test_abbreviation_is_no_sentence_end():
    chunks = split_for_qwen("Das ist z. B. ein Test. Zweiter Satz hier.", first_max=10)
    assert chunks[0].startswith("Das ist z. B. ein Test.")


def test_normalize_numbers_and_ordinals():
    out = normalize("Am 7. November um 14:30 Uhr kostet es 3,50 € bzw. 50%.")
    assert "siebten November" in out
    assert "vierzehn Uhr dreißig" in out
    assert "drei Euro fünfzig" in out
    assert "fünfzig Prozent" in out
