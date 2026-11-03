"""Download the public datasets / reference files used by the experiments.

    python scripts/fetch_data.py

* Tiny Shakespeare (public domain) - the experiment corpus
* GPT-2 ``encoder.json`` + ``vocab.bpe`` (MIT) - used to validate our BPE
  implementation against OpenAI's reference tokenizer

Files go to ``data/raw/`` which is git-ignored.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {
    "shakespeare.txt": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "gpt2_encoder.json": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
    "gpt2_vocab.bpe": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW / name
        if dest.exists():
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name}")
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
    print("done ->", RAW)


if __name__ == "__main__":
    main()
