"""Intrinsic metrics for comparing tokenizers.

All metrics are computed on a held-out text the tokenizer was *not* trained on,
because the interesting question is how a vocabulary generalises.

compression   ``chars_per_token`` / ``bytes_per_token`` - how much text one
              token carries on average. Higher = shorter sequences = cheaper
              attention (O(n^2)) and more context per window.
fertility     ``tokens_per_word`` - average tokens needed per word. A word-level
              model has 1.0 by definition; sub-word models trade >1.0 for an
              open vocabulary. (Term from Rust et al., 2021.)
unk_rate      fraction of emitted tokens that are the unknown token.
roundtrip     does ``decode(encode(x)) == x`` hold *exactly*?
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .base import Tokenizer
from .pretokenize import split_words

UNK_TOKENS = {"<unk>", "[UNK]"}


@dataclass
class Report:
    vocab_size: int
    n_tokens: int
    n_chars: int
    n_bytes: int
    n_words: int
    chars_per_token: float
    bytes_per_token: float
    fertility: float
    unk_rate: float
    roundtrip_exact: bool

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(tok: Tokenizer, text: str) -> Report:
    toks = tok.tokenize(text)
    n_tokens = max(len(toks), 1)
    n_words = max(len(split_words(text)), 1)
    n_bytes = len(text.encode("utf-8"))
    unk = sum(t in UNK_TOKENS for t in toks)
    return Report(
        vocab_size=tok.vocab_size,
        n_tokens=len(toks),
        n_chars=len(text),
        n_bytes=n_bytes,
        n_words=n_words,
        chars_per_token=len(text) / n_tokens,
        bytes_per_token=n_bytes / n_tokens,
        fertility=len(toks) / n_words,
        unk_rate=unk / n_tokens,
        roundtrip_exact=tok.decode(tok.encode(text)) == text,
    )


def oov_word_rate(train_text: str, test_text: str, lowercase: bool = True) -> float:
    """Fraction of test word *tokens* never seen in training (type-level OOV
    is also interesting but token-level is what hits a word-level model)."""
    norm = (lambda s: s.lower()) if lowercase else (lambda s: s)
    seen = set(map(norm, split_words(train_text)))
    test = list(map(norm, split_words(test_text)))
    return sum(w not in seen for w in test) / max(len(test), 1)
