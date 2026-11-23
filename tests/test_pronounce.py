from app import pronounce


RULES = "Mouzakrobat = Mausakrobat\n# Kommentar\nGaming = Geeming\nungültig ohne Gleich"


def test_apply_whole_words_case_insensitive():
    assert pronounce.apply("Hallo Mouzakrobat!", RULES) == "Hallo Mausakrobat!"
    assert pronounce.apply("mouzakrobat123", RULES) == "Mausakrobat123"
    assert pronounce.apply("Gaming-Abend", RULES) == "Geeming-Abend"
    assert pronounce.apply("Mouzakrobats", RULES) == "Mouzakrobats"  # kein ganzes Wort


def test_speakable_name_splits():
    assert pronounce.speakable_name("MouzAkrobat") == "Mouz Akrobat"
    assert pronounce.speakable_name("xXGamer99Xx") == "Gamer 99"
    assert pronounce.speakable_name("hans_peter") == "hans peter"
    assert pronounce.speakable_name("K3113rKind") == "K3113r Kind"
    assert pronounce.speakable_name("Max") == "Max"
