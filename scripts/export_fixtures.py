"""Export golden fixtures so the JavaScript port can be tested against Python.

    python scripts/export_fixtures.py     # writes web/tests/fixtures.json
    node --test web/tests/                # JS must reproduce every value

Python is the source of truth. For several (algorithm, corpus, vocab size)
configurations we record the learned vocab and merge list and the tokenization
of probe strings, plus the step-by-step encoding trace of a few words.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tokenlab import BPETokenizer, WordPieceTokenizer, WordTokenizer  # noqa: E402

CORPUS = (
    "low low low low low lower lower newest newest newest newest newest newest widest widest widest.\n"
    "The quick brown fox jumps over the lazy dog; the dog didn't care. It's 2024, isn't it?\n"
    "Tokenization splits text into tokens: un-believ-able, tokenizer, tokenizers, tokenized, tokenizing.\n"
    "Café naïve résumé — emoji 🐱 and 日本語のテキスト. Playing played player plays play walking walked.\n"
    "def f(x):\n    return x ** 2  # square\n"
)
SHAKE = (ROOT / "data" / "raw" / "shakespeare.txt")
if SHAKE.exists():
    CORPUS += SHAKE.read_text(encoding="utf-8")[:6000]

PROBES = [
    "lowest newer", "Hello, world!", "unbelievability", "  two  spaces ", "Tokenization of 12345 numbers.",
    "naïve café 🐱", "def g(y):\n\treturn y", "playing walker talks", "ZZZ qqq xyzzy", "",
]
TRACE_WORDS = ["lowest", " newest", "unbelievability", " tokenizers", "🐱"]


def main() -> None:
    out: dict = {"corpus": CORPUS, "probes": PROBES, "trace_words": TRACE_WORDS, "cases": []}

    for v in (290, 400):
        t = BPETokenizer.train(CORPUS, v, byte_level=True)
        out["cases"].append({
            "mode": "bpe-byte", "vocab_size": v, "merges": [list(m) for m in t.merges],
            "vocab": t.vocab, "tokens": [t.tokenize(p) for p in PROBES],
            "ids": [t.encode(p) for p in PROBES],
            "traces": {w: t.trace_word(w) for w in TRACE_WORDS},
        })
    for v in (60, 150):
        t = BPETokenizer.train(CORPUS, v, byte_level=False)
        out["cases"].append({
            "mode": "bpe-char", "vocab_size": v, "merges": [list(m) for m in t.merges],
            "vocab": t.vocab, "tokens": [t.tokenize(p) for p in PROBES],
            "ids": [t.encode(p) for p in PROBES],
            "traces": {w.strip(): t.trace_word(w.strip()) for w in TRACE_WORDS if w.strip().isascii()},
        })
    for v in (80, 200):
        t = WordPieceTokenizer.train(CORPUS, v)
        out["cases"].append({
            "mode": "wordpiece", "vocab_size": v, "vocab": t.vocab,
            "tokens": [t.tokenize(p) for p in PROBES], "ids": [t.encode(p) for p in PROBES],
        })
    t = WordTokenizer.train(CORPUS, 40)
    out["cases"].append({
        "mode": "word", "vocab_size": 40, "vocab": t.vocab,
        "tokens": [t.tokenize(p) for p in PROBES], "ids": [t.encode(p) for p in PROBES],
    })

    dest = ROOT / "web" / "tests" / "fixtures.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False))
    print(f"wrote {dest} ({dest.stat().st_size/1024:.0f} KB, {len(out['cases'])} cases)")


if __name__ == "__main__":
    main()
