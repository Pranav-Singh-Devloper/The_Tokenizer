"""Common interface shared by every tokenizer in the library."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path


class Tokenizer(ABC):
    """A tokenizer is a *pair of functions* ``encode: str -> [int]`` and
    ``decode: [int] -> str`` plus the vocabulary that gives the ids meaning."""

    #: filled in by subclasses: token string -> id
    vocab: dict[str, int]

    # ---- the contract -------------------------------------------------
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Text -> list of token *strings*."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Text -> list of token ids."""

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Token ids -> text."""

    @abstractmethod
    def to_dict(self) -> dict:
        """JSON-serialisable description of the trained tokenizer."""

    # ---- shared helpers ----------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def id_to_token(self) -> dict[int, str]:
        return {i: t for t, i in self.vocab.items()}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def __len__(self) -> int:
        return self.vocab_size


def build_vocab(specials: list[str], symbols: list[str]) -> dict[str, int]:
    """Assign ids: special tokens first, then ``symbols`` in the order given."""
    vocab: dict[str, int] = {}
    for tok in [*specials, *symbols]:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab
