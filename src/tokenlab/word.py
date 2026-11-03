"""Word-level tokenizer - the baseline every sub-word method is measured against.

Idea: one token per distinct word. The vocabulary is the ``max_size`` most
frequent words of the training corpus (optionally filtered by ``min_freq``).
Everything else collapses into a single ``<unk>`` token.

Strengths: short sequences, tokens are human-readable.
Weakness (the motivating problem of this whole project): the *open vocabulary
problem*. New words, typos, names, inflections ("tokenizing" when only
"tokenize" was seen) all become ``<unk>`` and their meaning is lost. Making
the vocabulary big enough to avoid this explodes the embedding matrix.
"""
from __future__ import annotations

from collections import Counter

from .base import Tokenizer, build_vocab
from .pretokenize import split_basic

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]


class WordTokenizer(Tokenizer):
    def __init__(self, vocab: dict[str, int], lowercase: bool = True):
        self.vocab = vocab
        self.lowercase = lowercase
        self._inv = {i: t for t, i in vocab.items()}

    # ---- training -----------------------------------------------------
    @classmethod
    def train(
        cls,
        corpus: str | list[str],
        max_size: int = 10_000,
        min_freq: int = 1,
        lowercase: bool = True,
    ) -> "WordTokenizer":
        text = corpus if isinstance(corpus, str) else "\n".join(corpus)
        counts = Counter(split_basic(text, lowercase))
        # most frequent first; ties broken alphabetically => deterministic
        ranked = sorted(
            (w for w, c in counts.items() if c >= min_freq),
            key=lambda w: (-counts[w], w),
        )
        room = max_size - len(SPECIALS)
        return cls(build_vocab(SPECIALS, ranked[: max(room, 0)]), lowercase)

    # ---- inference ----------------------------------------------------
    def tokenize(self, text: str) -> list[str]:
        """Words, with out-of-vocabulary words replaced by ``<unk>``."""
        return [w if w in self.vocab else UNK for w in split_basic(text, self.lowercase)]

    def encode(self, text: str) -> list[int]:
        return [self.vocab[t] for t in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        # lossy: whitespace and casing were discarded at pre-tokenization
        return " ".join(self._inv.get(i, UNK) for i in ids)

    # ---- (de)serialisation --------------------------------------------
    def to_dict(self) -> dict:
        return {"type": "word", "lowercase": self.lowercase, "vocab": self.vocab}

    @classmethod
    def from_dict(cls, d: dict) -> "WordTokenizer":
        return cls(d["vocab"], d.get("lowercase", True))
