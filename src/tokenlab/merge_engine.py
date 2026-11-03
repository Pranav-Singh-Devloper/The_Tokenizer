"""Incremental pair statistics shared by the BPE and WordPiece trainers.

Both algorithms follow the same loop::

    repeat until the vocabulary is big enough:
        1. find the best adjacent pair (a, b) in the corpus
        2. replace every occurrence of (a, b) by the new symbol ab
        3. record the merge

They differ ONLY in step 1 (the scoring rule): BPE picks the most *frequent*
pair, WordPiece picks the pair with the highest *likelihood gain*
``count(ab) / (count(a) * count(b))``.

A naive implementation re-counts every pair of the whole corpus after every
merge: O(merges x corpus). Here we keep

* ``pair_counts``  - weighted count of every adjacent pair,
* ``pair_words``   - inverted index pair -> set of words containing it,
* ``symbol_counts``- weighted count of every symbol (needed by WordPiece),

and a merge only touches the words that actually contain the merged pair.
"""
from __future__ import annotations

from collections import defaultdict

Pair = tuple[str, str]


class MergeEngine:
    def __init__(self, words: dict[tuple[str, ...], int]):
        """``words`` maps a symbol sequence to its corpus frequency."""
        self.seqs: list[list[str]] = []
        self.freqs: list[int] = []
        self.pair_counts: dict[Pair, int] = defaultdict(int)
        self.pair_words: dict[Pair, set[int]] = defaultdict(set)
        self.symbol_counts: dict[str, int] = defaultdict(int)
        for seq, freq in words.items():
            idx = len(self.seqs)
            self.seqs.append(list(seq))
            self.freqs.append(freq)
            for s in seq:
                self.symbol_counts[s] += freq
            for pair in zip(seq, seq[1:]):
                self.pair_counts[pair] += freq
                self.pair_words[pair].add(idx)

    # ---- step 1: choose ------------------------------------------------
    def best_by_frequency(self) -> Pair | None:
        """BPE rule. Ties broken by the lexicographically smallest pair."""
        if not self.pair_counts:
            return None
        top = max(self.pair_counts.values())
        return min(p for p, c in self.pair_counts.items() if c == top)

    def best_by_likelihood(self) -> Pair | None:
        """WordPiece rule: maximise count(ab) / (count(a) * count(b)).

        Fractions are compared by cross-multiplication on integers so that the
        result is exact and identical in every language implementation.
        """
        best: Pair | None = None
        best_num = best_den = 0
        for pair, c in self.pair_counts.items():
            den = self.symbol_counts[pair[0]] * self.symbol_counts[pair[1]]
            # c/den > best_num/best_den  <=>  c*best_den > best_num*den
            if best is None:
                better = True
            else:
                lhs, rhs = c * best_den, best_num * den
                better = lhs > rhs or (lhs == rhs and pair < best)
            if better:
                best, best_num, best_den = pair, c, den
        return best

    def best_by_gain(self) -> Pair | None:
        """Frequency-weighted likelihood gain: ``c * ln(c * N / (c_a * c_b))``.

        This is the first-order change in corpus log-likelihood under a unigram
        model when ``a b`` is replaced by one symbol (N = total symbol count).
        Unlike the bare ratio of :meth:`best_by_likelihood` it multiplies by
        ``c``, so a merge must be both *informative* and *frequent*. Uses
        floats, so it is only available in the Python implementation.
        """
        import math

        if not self.pair_counts:
            return None
        n = sum(self.symbol_counts.values())
        best: Pair | None = None
        best_g = -math.inf
        for pair, c in self.pair_counts.items():
            g = c * math.log(c * n / (self.symbol_counts[pair[0]] * self.symbol_counts[pair[1]]))
            if g > best_g or (g == best_g and best is not None and pair < best):
                best, best_g = pair, g
        return best

    # ---- step 2: apply -------------------------------------------------
    def merge(self, pair: Pair, new_symbol: str) -> int:
        """Apply ``pair -> new_symbol`` everywhere. Returns #words changed."""
        a, b = pair
        changed = 0
        # iterate over a snapshot: the index is mutated while we work
        for idx in sorted(self.pair_words.get(pair, ())):
            seq, freq = self.seqs[idx], self.freqs[idx]
            # left-to-right, non-overlapping replacement
            out: list[str] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    out.append(new_symbol)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            if len(out) == len(seq):  # pair no longer present (defensive)
                continue
            old_pairs = list(zip(seq, seq[1:]))
            new_pairs = list(zip(out, out[1:]))
            # 1. retract the word's old contribution ...
            for p in old_pairs:
                self.pair_counts[p] -= freq
                if self.pair_counts[p] <= 0:
                    del self.pair_counts[p]
            for s in seq:
                self.symbol_counts[s] -= freq
            # 2. ... and add the new one
            for p in new_pairs:
                self.pair_counts[p] += freq
            for s in out:
                self.symbol_counts[s] += freq
            # 3. keep the inverted index exact (no stale word ids)
            gone = set(old_pairs) - set(new_pairs)
            for p in gone:
                self._unindex(p, idx)
            for p in set(new_pairs):
                self.pair_words[p].add(idx)
            self.seqs[idx] = out
            changed += 1
        self.pair_counts.pop(pair, None)
        self.pair_words.pop(pair, None)
        return changed

    def _unindex(self, pair: Pair, idx: int) -> None:
        words = self.pair_words.get(pair)
        if words is not None:
            words.discard(idx)
            if not words:
                del self.pair_words[pair]

    # ---- debugging aid used by the tests ------------------------------
    def recount(self) -> tuple[dict[Pair, int], dict[str, int]]:
        """Recompute pair/symbol counts from scratch (O(corpus))."""
        pairs: dict[Pair, int] = defaultdict(int)
        syms: dict[str, int] = defaultdict(int)
        for seq, freq in zip(self.seqs, self.freqs):
            for s in seq:
                syms[s] += freq
            for p in zip(seq, seq[1:]):
                pairs[p] += freq
        return dict(pairs), dict(syms)
