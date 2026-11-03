"""WordPiece - the sub-word algorithm used by BERT and its descendants.

Origin: Schuster & Nakajima (2012), "Japanese and Korean Voice Search";
popularised by Wu et al. (2016, GNMT) and Devlin et al. (2018, BERT).

How it differs from BPE
-----------------------
TRAINING   Same merge loop, different *scoring rule*. BPE merges the most
           frequent pair. WordPiece merges the pair that most increases the
           likelihood of the training data under a unigram language model:

               score(a, b) = count(ab) / ( count(a) * count(b) )

           A frequent pair made of two very frequent symbols (e.g. "e","s")
           scores low; a pair whose parts rarely appear apart scores high. It
           favours *informative* merges over merely *common* ones.

ENCODING   There is no merge table replay. The final vocabulary alone is used
           with a greedy *longest-match-first* scan: take the longest prefix of
           the word that is in the vocabulary, emit it, continue from there
           with the ``##`` continuation prefix. If the scan gets stuck the
           whole word becomes ``[UNK]``.

Symbols that do not start a word carry the ``##`` prefix, so "playing" ->
``play ##ing``. This lets the vocabulary distinguish "ing" (a word) from
"##ing" (a suffix) and makes decoding trivial and unambiguous.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .base import Tokenizer, build_vocab
from .merge_engine import MergeEngine, Pair
from .pretokenize import split_basic

CONT = "##"
PAD, UNK, CLS, SEP, MASK = "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
SPECIALS = [PAD, UNK, CLS, SEP, MASK]
MAX_CHARS_PER_WORD = 100


@dataclass
class WPStep:
    step: int
    pair: Pair
    new_symbol: str
    count: int      # count(ab)
    count_a: int
    count_b: int
    score: float    # count / (count_a * count_b) -- display only
    vocab_size: int


def _initial_symbols(word: str) -> tuple[str, ...]:
    return tuple(c if i == 0 else CONT + c for i, c in enumerate(word))


def merged_symbol(a: str, b: str) -> str:
    return a + (b[len(CONT):] if b.startswith(CONT) else b)


class WordPieceTokenizer(Tokenizer):
    def __init__(self, vocab: dict[str, int], lowercase: bool = True):
        self.vocab = vocab
        self.lowercase = lowercase
        self._inv = {i: t for t, i in vocab.items()}

    # ---- training -----------------------------------------------------
    @classmethod
    def train(
        cls,
        corpus: str,
        vocab_size: int,
        lowercase: bool = True,
        on_merge: Callable[[WPStep], None] | None = None,
    ) -> "WordPieceTokenizer":
        counts = Counter(split_basic(corpus, lowercase))
        words = {_initial_symbols(w): c for w, c in counts.items()}
        alphabet = sorted({s for seq in words for s in seq})
        vocab = build_vocab(SPECIALS, alphabet)

        engine = MergeEngine(words)
        step = 0
        while len(vocab) < vocab_size:
            pair = engine.best_by_likelihood()
            if pair is None:
                break
            a, b = pair
            count = engine.pair_counts[pair]
            ca, cb = engine.symbol_counts[a], engine.symbol_counts[b]
            new_symbol = merged_symbol(a, b)
            engine.merge(pair, new_symbol)
            step += 1
            if new_symbol not in vocab:
                vocab[new_symbol] = len(vocab)
            if on_merge:
                on_merge(WPStep(step, pair, new_symbol, count, ca, cb, count / (ca * cb), len(vocab)))
        return cls(vocab, lowercase)

    # ---- inference ----------------------------------------------------
    def tokenize_word(self, word: str) -> list[str]:
        """Greedy longest-match-first (a.k.a. MaxMatch) on one word."""
        if len(word) > MAX_CHARS_PER_WORD:
            return [UNK]
        pieces: list[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = CONT + piece
                if piece in self.vocab:
                    found = piece
                    break
                end -= 1
            if found is None:
                return [UNK]  # whole word is unknown, as in BERT
            pieces.append(found)
            start = end
        return pieces

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for w in split_basic(text, self.lowercase):
            out.extend(self.tokenize_word(w))
        return out

    def encode(self, text: str) -> list[int]:
        unk = self.vocab[UNK]
        return [self.vocab.get(t, unk) for t in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        toks = [self._inv[i] for i in ids]
        words: list[str] = []
        for t in toks:
            if t.startswith(CONT) and words:
                words[-1] += t[len(CONT):]
            else:
                words.append(t)
        return " ".join(words)

    # ---- (de)serialisation --------------------------------------------
    def to_dict(self) -> dict:
        return {"type": "wordpiece", "lowercase": self.lowercase, "vocab": self.vocab}

    @classmethod
    def from_dict(cls, d: dict) -> "WordPieceTokenizer":
        return cls(d["vocab"], d.get("lowercase", True))
