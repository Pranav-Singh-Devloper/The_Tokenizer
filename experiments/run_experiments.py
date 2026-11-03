"""Reproduce every number and figure in docs/report.md.

    python scripts/fetch_data.py
    pip install matplotlib
    python experiments/run_experiments.py

Outputs (all deterministic):
    results/e1_vocab_sweep.csv      vocabulary-size sweep, all tokenizers
    results/e2_comparison.csv       head-to-head at a fixed budget
    results/e3_segmentations.json   how each model splits unseen words
    results/e4_domain_shift.csv     train on Shakespeare, test elsewhere
    results/summary.json            everything above (also copied to web/data/)
    results/figures/*.png
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tokenlab import BPETokenizer, WordPieceTokenizer, WordTokenizer  # noqa: E402
from tokenlab.gpt2 import load_gpt2  # noqa: E402
from tokenlab.metrics import evaluate, oov_word_rate  # noqa: E402
from tokenlab.pretokenize import split_words  # noqa: E402

RAW, OUT = ROOT / "data" / "raw", ROOT / "results"
FIG = OUT / "figures"
SWEEP = [300, 500, 1000, 2000, 4000, 8000, 16000]
FIXED = 2000

# --- out-of-domain probes for E4 (written for this project, no licensing issue)
MODERN = (
    "Large language models read text as sequences of tokens rather than words. "
    "The tokenizer decides how a sentence is split, so it directly affects cost, "
    "context length and how well a model handles rare words, numbers and typos. "
    "In 2023, researchers showed that tokenization quirks explain many odd failures."
)
CODE = (
    "def tokenize(text, vocab):\n    ids = []\n    for word in text.split():\n"
    "        ids.append(vocab.get(word, vocab['<unk>']))\n    return ids\n\n"
    "if __name__ == '__main__':\n    print(tokenize('hello world', {'<unk>': 0}))\n"
)
UNSEEN_WORDS = [
    "unbelievability", "tokenization", "Shakespearean", "counterrevolutionary",
    "overfitting", "antidisestablishmentarianism", "xyzzy", "naïveté",
]


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    corpus = (RAW / "shakespeare.txt").read_text(encoding="utf-8")
    cut = corpus.rfind("\n", 0, int(len(corpus) * 0.9)) + 1
    train, test = corpus[:cut], corpus[cut:]
    print(f"train={len(train):,} chars  test={len(test):,} chars")
    summary: dict = {
        "corpus": {"name": "Tiny Shakespeare", "train_chars": len(train), "test_chars": len(test),
                   "train_unique_words": len(set(map(str.lower, split_words(train)))),
                   "test_oov_word_rate": oov_word_rate(train, test)},
        "sweep_sizes": SWEEP,
    }

    # ---------------- E1: vocabulary-size sweep ------------------------
    print("E1 training the largest model of each family once (nested vocabularies)...")
    max_v = max(SWEEP)
    (bpe_byte, t1) = timed(BPETokenizer.train, train, max_v, True)
    (bpe_cls, t2) = timed(BPETokenizer.train, train, max_v, False)
    (wp, t3) = timed(WordPieceTokenizer.train, train, max_v)
    (wordtok, t4) = timed(WordTokenizer.train, train, max_v)
    (wpg, t5) = timed(WordPieceTokenizer.train, train, max_v, score="gain")
    print(f"  train time  byte-BPE {t1:.1f}s  char-BPE {t2:.1f}s  WordPiece {t3:.1f}s  "
          f"WordPiece-gain {t5:.1f}s  word {t4:.2f}s")
    summary["train_seconds_at_max_vocab"] = {"byte_bpe": t1, "char_bpe": t2, "wordpiece": t3,
                                             "wordpiece_gain": t5, "word": t4}

    def word_at(v):  # word vocab nested by frequency rank as well
        return WordTokenizer(
            {t: i for t, i in wordtok.vocab.items() if i < v}, wordtok.lowercase
        )

    families = {
        "word": word_at,
        "char-BPE": bpe_cls.truncated,
        "byte-BPE": bpe_byte.truncated,
        "WordPiece": wp.truncated,
        "WordPiece-gain": wpg.truncated,
    }
    rows = []
    for name, make in families.items():
        for v in SWEEP:
            tok = make(v)
            r = evaluate(tok, test).as_dict()
            rows.append({"tokenizer": name, "target_vocab": v, **r})
            print(f"  {name:10s} V={tok.vocab_size:6d} chars/tok={r['chars_per_token']:.2f} "
                  f"fertility={r['fertility']:.2f} unk={r['unk_rate']*100:.2f}%")
    write_csv(OUT / "e1_vocab_sweep.csv", rows)
    summary["e1"] = rows

    # ---------------- E2: head-to-head at a fixed budget -----------------
    print(f"E2 head-to-head at V={FIXED}")
    e2 = []
    for name, make, secs in [("word", word_at, t4), ("char-BPE", bpe_cls.truncated, None),
                             ("byte-BPE", bpe_byte.truncated, None), ("WordPiece", wp.truncated, None),
                             ("WordPiece-gain", wpg.truncated, None)]:
        tok = make(FIXED)
        # honest training time for THIS vocab size
        if name == "char-BPE":
            _, secs = timed(BPETokenizer.train, train, FIXED, False)
        elif name == "byte-BPE":
            _, secs = timed(BPETokenizer.train, train, FIXED, True)
        elif name == "WordPiece":
            _, secs = timed(WordPieceTokenizer.train, train, FIXED)
        elif name == "WordPiece-gain":
            _, secs = timed(WordPieceTokenizer.train, train, FIXED, score="gain")
        r = evaluate(tok, test).as_dict()
        e2.append({"tokenizer": name, "train_seconds": round(secs, 2), **r})
    write_csv(OUT / "e2_comparison.csv", e2)
    summary["e2"] = e2

    # ---------------- E3: segmentation of unseen words ---------------------
    models = {"word": word_at(FIXED), "char-BPE": bpe_cls.truncated(FIXED),
              "byte-BPE": bpe_byte.truncated(FIXED), "WordPiece": wp.truncated(FIXED),
              "WordPiece-gain": wpg.truncated(FIXED)}
    e3 = {w: {n: m.tokenize(" " + w if n == "byte-BPE" else w) for n, m in models.items()}
          for w in UNSEEN_WORDS}
    (OUT / "e3_segmentations.json").write_text(json.dumps(e3, ensure_ascii=False, indent=1))
    summary["e3"] = e3

    # ---------------- E4: domain shift ---------------------------------
    print("E4 domain shift")
    probes = {"Shakespeare (in-domain test)": test, "Modern prose": MODERN, "Python code": CODE}
    e4 = []
    ref = None
    if (RAW / "gpt2_encoder.json").exists():
        ref = load_gpt2(RAW / "gpt2_encoder.json", RAW / "gpt2_vocab.bpe")
    for dom, text in probes.items():
        for name in models:
            r = evaluate(models[name], text)
            e4.append({"domain": dom, "tokenizer": f"{name} (V={FIXED}, Shakespeare-trained)",
                       "bytes_per_token": round(r.bytes_per_token, 3), "unk_rate": round(r.unk_rate, 4)})
        if ref:
            r = evaluate(ref, text)
            e4.append({"domain": dom, "tokenizer": "GPT-2 byte-BPE (V=50257, WebText-trained)",
                       "bytes_per_token": round(r.bytes_per_token, 3), "unk_rate": round(r.unk_rate, 4)})
    write_csv(OUT / "e4_domain_shift.csv", e4)
    summary["e4"] = e4

    # ---------------- Zipf data for the figure ---------------------------
    wc = Counter(map(str.lower, split_words(train)))
    bc = Counter(bpe_byte.truncated(FIXED).tokenize(train))
    summary["zipf"] = {"word": sorted(wc.values(), reverse=True)[:3000],
                       "byte-BPE": sorted(bc.values(), reverse=True)[:3000]}

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False))
    (ROOT / "web" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "web" / "data" / "experiments.json").write_text(json.dumps(summary, ensure_ascii=False))
    make_figures(summary)
    print("done ->", OUT)


def make_figures(s: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: skipping figures")
        return
    colors = {"word": "#d95f02", "char-BPE": "#1b9e77", "byte-BPE": "#7570b3", "WordPiece": "#e7298a",
              "WordPiece-gain": "#66a61e"}
    rows = s["e1"]

    def series(name, key):
        return [r["target_vocab"] for r in rows if r["tokenizer"] == name], [r[key] for r in rows if r["tokenizer"] == name]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for key, a, title, pct in [("chars_per_token", ax[0], "Compression (chars / token) ↑", 1),
                               ("fertility", ax[1], "Fertility (tokens / word) ↓", 1),
                               ("unk_rate", ax[2], "Unknown-token rate (%) ↓", 100)]:
        for name in colors:
            x, y = series(name, key)
            a.plot(x, [v * pct for v in y], marker="o", color=colors[name], label=name)
        a.set_xscale("log"); a.set_xlabel("vocabulary size"); a.set_title(title); a.grid(alpha=.3)
    ax[0].legend()
    fig.suptitle("E1 - vocabulary-size sweep on held-out Shakespeare")
    fig.tight_layout(); fig.savefig(FIG / "e1_vocab_sweep.png", dpi=140); plt.close(fig)

    fig, a = plt.subplots(figsize=(6, 4.2))
    for name, ys in s["zipf"].items():
        a.loglog(range(1, len(ys) + 1), ys, label=name, color=colors[name if name in colors else "byte-BPE"])
    a.set_xlabel("rank"); a.set_ylabel("frequency"); a.legend(); a.grid(alpha=.3, which="both")
    a.set_title("E5 - token frequency vs rank (Zipf)")
    fig.tight_layout(); fig.savefig(FIG / "e5_zipf.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
