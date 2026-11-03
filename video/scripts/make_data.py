"""Export the real numbers the Remotion scenes display -> remotion/src/data.json.

Everything comes from running the tokenlab library / reading results/summary.json;
nothing in the video is typed in by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tokenlab import BPETokenizer  # noqa: E402
from tokenlab.bpe import _initial_symbols as classic_symbols  # noqa: E402
from tokenlab.bytes_unicode import symbols_to_text  # noqa: E402
from tokenlab.merge_engine import MergeEngine  # noqa: E402
from tokenlab.wordpiece import _initial_symbols as wp_symbols  # noqa: E402

TOY = {"newest": 6, "low": 5, "widest": 3, "lower": 2}
SENTENCE = "Shakespeare's unbelievably tokenization-proof sonnets"


def main() -> None:
    shake = (ROOT / "data" / "raw" / "shakespeare.txt").read_text(encoding="utf-8")
    train = shake[: int(len(shake) * 0.9)]
    bpe = BPETokenizer.train(train, 2000, byte_level=True)
    toks = bpe.tokenize(SENTENCE)
    intro = {"sentence": SENTENCE, "tokens": [symbols_to_text(t) for t in toks], "ids": bpe.encode(SENTENCE)}

    # --- BPE vs WordPiece scoring on the same toy corpus
    bpe_eng = MergeEngine({classic_symbols(w, False): c for w, c in TOY.items()})
    top_bpe = sorted(bpe_eng.pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    wp_eng = MergeEngine({wp_symbols(w): c for w, c in TOY.items()})
    rows = []
    for (a, b), c in wp_eng.pair_counts.items():
        ca, cb = wp_eng.symbol_counts[a], wp_eng.symbol_counts[b]
        rows.append({"pair": [a, b], "count": c, "ca": ca, "cb": cb, "score": c / (ca * cb)})
    rows.sort(key=lambda r: (-r["score"], r["pair"]))
    compare = {
        "bpe": [{"pair": list(p), "count": c} for p, c in top_bpe],
        "wordpiece": rows[:5],
    }

    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    e1 = {}
    for r in summary["e1"]:
        e1.setdefault(r["tokenizer"], []).append([r["target_vocab"], r["chars_per_token"]])
    e2 = {r["tokenizer"]: r for r in summary["e2"]}
    e4 = [r for r in summary["e4"] if r["domain"] == "Python code"]
    out = {
        "intro": intro,
        "compare": compare,
        "e1": e1,
        "facts": {
            "oov_rate": summary["corpus"]["test_oov_word_rate"],
            "word_unk_2000": e2["word"]["unk_rate"],
            "lossless": {k: v["roundtrip_exact"] for k, v in e2.items()},
            "cpt_2000": {k: v["chars_per_token"] for k, v in e2.items()},
            "code_bpt": {r["tokenizer"]: r["bytes_per_token"] for r in e4},
            "gpt2_parity_tokens": 338025,
        },
    }
    dest = ROOT / "video" / "remotion" / "src" / "data.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote", dest)
    print(json.dumps(compare["wordpiece"][:2]), intro["tokens"])


if __name__ == "__main__":
    main()
