"""Our BPE encoder must reproduce OpenAI's GPT-2 tokenizer exactly.

Skipped automatically unless ``tiktoken`` is installed and
``python scripts/fetch_data.py`` has been run.
"""
from pathlib import Path

import pytest

tiktoken = pytest.importorskip("tiktoken")

from tokenlab.gpt2 import load_gpt2  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
pytestmark = pytest.mark.skipif(
    not (RAW / "gpt2_encoder.json").exists(), reason="run scripts/fetch_data.py first"
)

CASES = [
    "", "Hello, world!", "it's we're they've I'm you'll he'd", "  leading and trailing  ",
    "tab\tseparated", "x_y__z", "3.14159 and 1,000,000", "naïve café", "日本語のテキスト",
    "🐱🐶 emoji 🚀", "def f(x):\n    return x**2\n", "\r\n windows\r\n", "a" * 300, "...!!!???",
]


@pytest.fixture(scope="module")
def pair():
    return (
        load_gpt2(RAW / "gpt2_encoder.json", RAW / "gpt2_vocab.bpe"),
        tiktoken.get_encoding("gpt2"),
    )


@pytest.mark.parametrize("text", CASES)
def test_identical_ids(pair, text):
    ours, ref = pair
    assert ours.encode(text) == ref.encode(text)
    assert ours.decode(ours.encode(text)) == text


def test_identical_on_shakespeare_sample(pair):
    ours, ref = pair
    text = (RAW / "shakespeare.txt").read_text(encoding="utf-8")[:200_000]
    assert ours.encode(text) == ref.encode(text)
