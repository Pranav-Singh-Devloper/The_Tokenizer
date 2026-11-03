import random

from tokenlab.merge_engine import MergeEngine


def _words(rng):
    letters = "abcd"
    return {
        tuple(rng.choice(letters) for _ in range(rng.randint(1, 8))): rng.randint(1, 20)
        for _ in range(40)
    }


def test_incremental_counts_match_recount_after_every_merge():
    """The core invariant: the O(touched words) update equals an O(corpus) recount."""
    rng = random.Random(1)
    for trial in range(15):
        eng = MergeEngine(_words(rng))
        for _ in range(12):
            pair = eng.best_by_frequency()
            if pair is None:
                break
            eng.merge(pair, pair[0] + pair[1])
            pairs, syms = eng.recount()
            assert dict(eng.pair_counts) == pairs
            assert {k: v for k, v in eng.symbol_counts.items() if v} == syms
            # inverted index exactness: no stale ids, no missing ids
            for p, idxs in eng.pair_words.items():
                for i in idxs:
                    assert p in set(zip(eng.seqs[i], eng.seqs[i][1:]))
            for i, seq in enumerate(eng.seqs):
                for p in zip(seq, seq[1:]):
                    assert i in eng.pair_words[p]


def test_overlapping_pair_is_replaced_left_to_right():
    eng = MergeEngine({("a", "a", "a"): 1, ("a", "a", "a", "a"): 1})
    eng.merge(("a", "a"), "aa")
    assert sorted(map(tuple, eng.seqs)) == [("aa", "a"), ("aa", "aa")]


def test_tie_break_is_lexicographic():
    eng = MergeEngine({("b", "c"): 3, ("a", "z"): 3})
    assert eng.best_by_frequency() == ("a", "z")


def test_likelihood_prefers_rare_parts_over_frequent_pair():
    # ("e","s") is the most frequent pair, but e and s are everywhere.
    # ("q","u") is rarer in absolute terms but q and u almost only occur together.
    words = {("e", "s"): 50, ("e", "x"): 50, ("s", "x"): 50, ("q", "u"): 10}
    eng = MergeEngine(words)
    assert eng.best_by_frequency() == ("e", "s")
    assert eng.best_by_likelihood() == ("q", "u")


def test_gain_weights_by_frequency():
    # same ratio as ("q","u") but 100x more frequent -> must win under 'gain'
    words = {("q", "u"): 1, ("z", "k"): 100, ("e", "s"): 50, ("e", "x"): 50, ("s", "x"): 50}
    eng = MergeEngine(words)
    assert eng.best_by_gain() == ("z", "k")
