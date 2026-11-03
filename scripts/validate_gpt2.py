"""Validate our BPE encoder against OpenAI's reference GPT-2 tokenizer.

    pip install tiktoken      # reference implementation (validation only)
    python scripts/fetch_data.py
    python scripts/validate_gpt2.py

Loads GPT-2's 50,000 merges into *our* tokenizer, encodes Tiny Shakespeare plus
a set of adversarial strings with both implementations, and compares ids.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tiktoken  # noqa: E402

from tokenlab.gpt2 import load_gpt2  # noqa: E402

RAW = ROOT / "data" / "raw"
EDGE = [
    "", " ", "  ", "\n\n\n", "Hello, world!", "it's we're they've I'm you'll he'd",
    "  leading and trailing  ", "tab\tseparated\tvalues", "x_y__z", "3.14159 and 1,000,000",
    "naïve café résumé", "日本語のテキスト", "🐱🐶 emoji 🚀", "def f(x):\n    return x**2\n",
    "<|endoftext|>", "mixed CASE and UPPER", "a" * 200, "...!!!???", "\r\n windows line\r\n",
]


def main() -> int:
    ours = load_gpt2(RAW / "gpt2_encoder.json", RAW / "gpt2_vocab.bpe")
    ref = tiktoken.get_encoding("gpt2")
    texts = EDGE + [(RAW / "shakespeare.txt").read_text(encoding="utf-8")]
    bad = 0
    for t in texts:
        t0 = time.perf_counter()
        a = ours.encode(t)
        t1 = time.perf_counter()
        b = ref.encode(t, allowed_special={"<|endoftext|>"})
        t2 = time.perf_counter()
        ok = a == b and ours.decode(a) == t
        bad += not ok
        label = (t[:28] + "…") if len(t) > 28 else t
        print(f"{'OK ' if ok else 'FAIL'} {len(a):>7} tokens  ours={t1-t0:6.2f}s ref={t2-t1:6.3f}s  {label!r}")
    print("\nALL IDENTICAL" if not bad else f"\n{bad} MISMATCHES")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
