import json
import random

import pytest

from tokenlab import BPETokenizer, load

SENNRICH = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3


def test_sennrich_example_first_merges():
    """Reproduces the toy example from Sennrich et al. (2016), Fig. 1."""
    steps = []
    BPETokenizer.train(SENNRICH, 40, byte_level=False, on_merge=steps.append)
    assert [s.pair for s in steps[:2]] == [("e", "s"), ("es", "t</w>")]
    assert steps[0].count == 9  # newest(6) + widest(3)


def test_classic_mode_encodes_unseen_word_into_known_pieces():
    t = BPETokenizer.train(SENNRICH, 40, byte_level=False)
    toks = t.tokenize("lowest")
    assert "".join(toks).replace("</w>", "") == "lowest"
    assert len(toks) < len("lowest")  # compressed below char level


def test_classic_unknown_char_becomes_unk():
    t = BPETokenizer.train(SENNRICH, 40, byte_level=False)
    assert t.encode("zzz")[0] == t.vocab["<unk>"]


def test_byte_level_roundtrip_any_string():
    t = BPETokenizer.train("the cat sat on the mat. " * 20, 300)
    rng = random.Random(7)
    pool = "abc def\n\t.,é日🐱\x00 ‮"
    for _ in range(300):
        s = "".join(rng.choice(pool) for _ in range(rng.randint(0, 30)))
        assert t.decode(t.encode(s)) == s


def test_byte_level_never_unk():
    t = BPETokenizer.train("aaa", 270)
    assert t.vocab["<|endoftext|>"] not in t.encode("🐱日本語 anything")


def test_special_tokens_are_atomic():
    t = BPETokenizer.train("hello world " * 10, 280)
    ids = t.encode("hello<|endoftext|>world")
    assert ids.count(t.vocab["<|endoftext|>"]) == 1
    assert t.decode(ids) == "hello<|endoftext|>world"
    # when not allowed they are just text
    assert t.vocab["<|endoftext|>"] not in t.encode("<|endoftext|>", allow_special=False)


def test_training_is_deterministic():
    a = BPETokenizer.train(SENNRICH * 3, 60, byte_level=False)
    b = BPETokenizer.train(SENNRICH * 3, 60, byte_level=False)
    assert a.merges == b.merges and a.vocab == b.vocab


def test_vocab_size_is_respected_and_ids_are_dense():
    t = BPETokenizer.train("abracadabra " * 50, 280)
    assert t.vocab_size <= 280
    assert sorted(t.vocab.values()) == list(range(t.vocab_size))


def test_truncated_equals_shorter_training_run():
    corpus = "the quick brown fox jumps over the lazy dog. " * 30
    big = BPETokenizer.train(corpus, 330)
    small = BPETokenizer.train(corpus, 290)
    cut = big.truncated(290)
    assert cut.merges == small.merges
    assert cut.vocab == small.vocab
    assert cut.encode("the lazy fox") == small.encode("the lazy fox")


def test_more_merges_never_lengthen_encoding():
    corpus = "the quick brown fox jumps over the lazy dog. " * 30
    big = BPETokenizer.train(corpus, 340)
    lens = [len(big.truncated(v).encode(corpus)) for v in (260, 280, 300, 340)]
    assert lens == sorted(lens, reverse=True)


def test_save_load_roundtrip(tmp_path):
    t = BPETokenizer.train(SENNRICH, 50, byte_level=False)
    p = tmp_path / "bpe.json"
    t.save(p)
    assert json.loads(p.read_text())["type"] == "bpe"
    u = load(p)
    assert u.merges == t.merges and u.encode("newest lower") == t.encode("newest lower")


def test_trace_word_ends_at_encoding():
    t = BPETokenizer.train("the cat sat on the mat. " * 20, 300)
    for w in ["the", " cat", " unseenword"]:
        states = t.trace_word(w)
        assert states[-1]["symbols"] == list(t._merge_symbols(tuple(states[0]["symbols"])))
        ranks = [s["rank"] for s in states[1:]]
        assert ranks == sorted(ranks)  # merges are applied in increasing rank
