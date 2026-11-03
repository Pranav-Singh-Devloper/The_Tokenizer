"""Pre-tokenization: the step that runs *before* any learned sub-word model.

A sub-word model never sees raw text. Text is first cut into *pre-tokens*
("words" in the loose sense) and the learned merges are only ever applied
inside a single pre-token. This is a design decision with real consequences:
it stops the model from learning merges that straddle word boundaries and it
makes training tractable (we count unique pre-tokens, not the whole corpus).

Three splitters are provided:

``split_words``
    ``\\w+`` runs and single punctuation marks. Whitespace is *discarded*, so
    the mapping text -> tokens is lossy (used by the word-level, classic BPE
    and WordPiece models).

``split_basic``
    BERT-style basic tokenizer: optional lower-casing + accent stripping, then
    words and punctuation.

``split_gpt2``
    The GPT-2 regular expression. Whitespace is *kept* (attached as a leading
    space on the following word), so ``"".join(pieces) == text`` always holds.
    This is what makes byte-level BPE lossless.
"""
from __future__ import annotations

import re
import unicodedata

# ``\p{L}`` / ``\p{N}`` are not available in the stdlib ``re`` module, so we use
# the standard equivalents: letters == ``[^\W\d_]`` and "other" == ``[^\s\w]|_``.
# See docs/report.md ("Known deviations") for the (tiny) differences.
GPT2_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?(?:[^\s\w]|_)+"
    r"|\s+(?!\S)"
    r"|\s+"
)

WORD_PATTERN = re.compile(r"\w+|[^\w\s]")


def split_words(text: str) -> list[str]:
    """Words and punctuation; whitespace is dropped."""
    return WORD_PATTERN.findall(text)


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def split_basic(text: str, lowercase: bool = True) -> list[str]:
    """BERT-style basic tokenization (optional lowercase + accent strip)."""
    if lowercase:
        text = strip_accents(text.lower())
    return split_words(text)


def split_gpt2(text: str) -> list[str]:
    """GPT-2 pre-tokenization. Lossless: ``"".join(result) == text``."""
    return GPT2_PATTERN.findall(text)
