"""tokenlab - tokenizers from scratch (word, BPE, byte-level BPE, WordPiece)."""

__version__ = "0.1.0"

from .base import Tokenizer
from .bpe import BPETokenizer
from .word import WordTokenizer
from .wordpiece import WordPieceTokenizer

__all__ = ["Tokenizer", "WordTokenizer", "BPETokenizer", "WordPieceTokenizer"]


def load(path):
    """Load any tokenizer saved with ``Tokenizer.save``."""
    import json
    from pathlib import Path

    d = json.loads(Path(path).read_text(encoding="utf-8"))
    kinds = {"word": WordTokenizer, "bpe": BPETokenizer, "wordpiece": WordPieceTokenizer}
    return kinds[d["type"]].from_dict(d)
