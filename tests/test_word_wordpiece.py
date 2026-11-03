from tokenlab import WordPieceTokenizer, WordTokenizer

CORPUS = "the cat sat on the mat . the dog sat on the log . playing played player plays play"


def test_word_vocab_ranking_and_unk():
    t = WordTokenizer.train(CORPUS, max_size=10)
    assert t.vocab["<pad>"] == 0 and t.vocab["<unk>"] == 1
    assert t.vocab["the"] < t.vocab["cat"]  # frequent first
    assert t.encode("cat zebra")[1] == t.vocab["<unk>"]
    assert t.vocab_size <= 10


def test_word_min_freq():
    t = WordTokenizer.train(CORPUS, min_freq=2)
    assert "the" in t.vocab and "zebra" not in t.vocab and "cat" not in t.vocab


def test_word_lowercase_flag():
    assert WordTokenizer.train("The the", lowercase=True).tokenize("THE") == ["the"]
    assert WordTokenizer.train("The the", lowercase=False).tokenize("THE") == ["THE"]


def test_wordpiece_greedy_longest_match():
    t = WordPieceTokenizer({"[UNK]": 0, "un": 1, "##aff": 2, "##able": 3, "##a": 4, "##ff": 5}, True)
    assert t.tokenize_word("unaffable") == ["un", "##aff", "##able"]


def test_wordpiece_unk_for_whole_word():
    t = WordPieceTokenizer.train(CORPUS, 40)
    assert t.tokenize("zebra") == ["[UNK]"]  # 'z','b' never seen: whole word unknown


def test_wordpiece_roundtrip_on_seen_words():
    t = WordPieceTokenizer.train(CORPUS, 60)
    s = "the cat played on the mat"
    assert t.decode(t.encode(s)) == s


def test_wordpiece_continuation_prefix():
    t = WordPieceTokenizer.train(CORPUS, 60)
    toks = t.tokenize("playing")
    assert all(x.startswith("##") for x in toks[1:])
    assert "".join(x.removeprefix("##") for x in toks) == "playing"


def test_wordpiece_trainer_step_scores():
    steps = []
    WordPieceTokenizer.train(CORPUS, 45, on_merge=steps.append)
    assert steps and all(abs(s.score - s.count / (s.count_a * s.count_b)) < 1e-12 for s in steps)
