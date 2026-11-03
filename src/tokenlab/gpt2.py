"""Load OpenAI's published GPT-2 tokenizer into our own :class:`BPETokenizer`.

This is a *validation* tool, not a shortcut: the encoder/decoder used is the
one we wrote from scratch in ``bpe.py``; only the data (merge list + ids) is
OpenAI's. If our implementation reproduces ``tiktoken``'s GPT-2 output exactly
on a large text, that is strong evidence the algorithm is implemented
correctly. See ``tests/test_gpt2_parity.py`` and ``scripts/validate_gpt2.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

from .bpe import BPETokenizer


def load_gpt2(encoder_json: str | Path, vocab_bpe: str | Path) -> BPETokenizer:
    vocab = json.loads(Path(encoder_json).read_text(encoding="utf-8"))
    lines = Path(vocab_bpe).read_text(encoding="utf-8").split("\n")
    # first line is a "#version" header, last line is empty
    merges = [tuple(l.split(" ")) for l in lines[1:] if l]
    return BPETokenizer(vocab, merges, byte_level=True, specials=["<|endoftext|>"])
