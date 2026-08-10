from detector.data.normalize import normalize_text


def test_strips_zero_width() -> None:
    attacked = "The​ qui​ck brown fox"
    assert normalize_text(attacked) == "The quick brown fox"


def test_maps_cyrillic_homoglyphs() -> None:
    # 'о' and 'е' below are Cyrillic
    attacked = "Thе quick brоwn fox"
    assert normalize_text(attacked) == "The quick brown fox"


def test_nfkc_fullwidth() -> None:
    assert normalize_text("Ｈello") == "Hello"


def test_plain_text_unchanged() -> None:
    s = "An ordinary sentence, with punctuation — and a dash."
    assert normalize_text(s) == s
