import random

from tokenlab.bytes_unicode import bytes_to_unicode, symbols_to_text, text_to_symbols
from tokenlab.pretokenize import split_basic, split_gpt2, split_words


def test_gpt2_split_is_lossless():
    for t in ["Hello, world!  It's 2024... naïve café\n\nx_y", "", "   ", "a\tb\r\nc", "🐱 emoji 日本語"]:
        assert "".join(split_gpt2(t)) == t


def test_gpt2_split_shape():
    assert split_gpt2("Hello, world!") == ["Hello", ",", " world", "!"]
    assert split_gpt2("it's") == ["it", "'s"]
    assert split_gpt2("a  b") == ["a", " ", " b"]  # run of spaces: last one sticks to the word


def test_gpt2_split_lossless_random():
    rng = random.Random(0)
    alphabet = "abcXYZ019 \n\t.,!?'_é日🐱"
    for _ in range(300):
        t = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert "".join(split_gpt2(t)) == t


def test_split_words_and_basic():
    assert split_words("Hi, you!") == ["Hi", ",", "you", "!"]
    assert split_basic("Café, NAÏVE") == ["cafe", ",", "naive"]
    assert split_basic("Café", lowercase=False) == ["Café"]


def test_byte_map_is_bijection():
    m = bytes_to_unicode()
    assert len(m) == 256 and len(set(m.values())) == 256
    assert m[ord(" ")] == "Ġ"
    assert m[ord("a")] == "a"


def test_byte_roundtrip_all_unicode_planes():
    s = "aé日🐱\x00\x7f‮"
    assert symbols_to_text("".join(text_to_symbols(s))) == s
