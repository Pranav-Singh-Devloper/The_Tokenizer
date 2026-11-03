"""Byte Pair Encoding (BPE) - the sub-word algorithm behind GPT-2/3/4, Llama, ...

Origin: Gage (1994) as a data-compression scheme; adapted to NLP by
Sennrich, Haddow & Birch (2016), "Neural Machine Translation of Rare Words with
Subword Units"; made byte-level by Radford et al. (2019, GPT-2).

TRAINING (learn a merge table)
    1. Pre-tokenize the corpus and count the unique pre-tokens.
    2. Start with every pre-token written as a sequence of base symbols
       (characters, or bytes in byte-level mode).
    3. Repeatedly merge the most frequent adjacent pair into a new symbol and
       append the rule to the ordered merge list.
    4. Stop when the vocabulary reaches ``vocab_size``.

ENCODING (apply the merge table to unseen text)
    For each pre-token, start from base symbols and repeatedly apply the
    *earliest-learned* merge that is applicable, until none applies. The merge
    order (rank) *is* the model: it replays training on new data.

Two flavours are supported by one class:

``byte_level=False``  classic Sennrich BPE: characters + an end-of-word marker
                      ``</w>``; characters never seen in training become
                      ``<unk>``; decoding is lossy (whitespace is normalised).
``byte_level=True``   GPT-2 style: UTF-8 bytes, whitespace kept inside
                      pre-tokens. Nothing is ever unknown and
                      ``decode(encode(x)) == x`` for *every* string.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

from .base import Tokenizer, build_vocab
from .bytes_unicode import bytes_to_unicode, symbols_to_text, text_to_symbols
from .merge_engine import MergeEngine, Pair
from .pretokenize import split_gpt2, split_words

EOW = "</w>"  # end-of-word marker (classic mode only)
CLASSIC_SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
BYTE_SPECIALS = ["<|endoftext|>"]
_INF = 1 << 60


@dataclass
class MergeStep:
    """One training iteration - this is what the web visualizer replays."""

    step: int
    pair: Pair
    new_symbol: str
    count: int          # corpus frequency of the pair at merge time
    words_changed: int  # unique pre-tokens rewritten
    vocab_size: int     # vocabulary size after the merge


def _initial_symbols(token: str, byte_level: bool) -> tuple[str, ...]:
    if byte_level:
        return tuple(text_to_symbols(token))
    chars = list(token)
    chars[-1] += EOW
    return tuple(chars)


class BPETokenizer(Tokenizer):
    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[Pair],
        byte_level: bool = True,
        specials: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = [tuple(m) for m in merges]
        self.byte_level = byte_level
        self.specials = (
            list(specials)
            if specials is not None
            else (BYTE_SPECIALS if byte_level else CLASSIC_SPECIALS)
        )
        self.ranks: dict[Pair, int] = {m: i for i, m in enumerate(self.merges)}
        self._inv = {i: t for t, i in vocab.items()}
        self._cache: dict[str, tuple[str, ...]] = {}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(s) for s in sorted(self.specials, key=len, reverse=True)) + ")")
            if self.specials
            else None
        )

    # ==================================================================
    # training
    # ==================================================================
    @classmethod
    def train(
        cls,
        corpus: str | Iterable[str],
        vocab_size: int,
        byte_level: bool = True,
        min_frequency: int = 1,
        on_merge: Callable[[MergeStep], None] | None = None,
        specials: list[str] | None = None,
    ) -> "BPETokenizer":
        text = corpus if isinstance(corpus, str) else "\n".join(corpus)
        specials = (
            list(specials)
            if specials is not None
            else (BYTE_SPECIALS if byte_level else CLASSIC_SPECIALS)
        )
        # 1. pre-tokenize + count unique pre-tokens
        split = split_gpt2 if byte_level else split_words
        counts = Counter(split(text))
        words = {_initial_symbols(w, byte_level): c for w, c in counts.items()}

        # 2. base alphabet
        if byte_level:
            base = [bytes_to_unicode()[b] for b in range(256)]
        else:
            base = sorted({s for seq in words for s in seq})
        vocab = build_vocab(specials, base)

        # 3. merge loop
        engine = MergeEngine(words)
        merges: list[Pair] = []
        while len(vocab) < vocab_size:
            pair = engine.best_by_frequency()
            if pair is None or engine.pair_counts[pair] < min_frequency:
                break
            count = engine.pair_counts[pair]
            new_symbol = pair[0] + pair[1]
            changed = engine.merge(pair, new_symbol)
            merges.append(pair)
            if new_symbol not in vocab:  # two merge paths can yield one string
                vocab[new_symbol] = len(vocab)
            if on_merge:
                on_merge(MergeStep(len(merges), pair, new_symbol, count, changed, len(vocab)))
        return cls(vocab, merges, byte_level, specials)

    def truncated(self, vocab_size: int) -> "BPETokenizer":
        """The tokenizer you would have got by stopping training earlier.

        BPE vocabularies are *nested*: the first k merges of a size-V model are
        exactly the size-(base+k) model. This lets experiments sweep vocabulary
        sizes from a single training run.
        """
        merges: list[Pair] = []
        for m in self.merges:
            tok = m[0] + m[1]
            if self.vocab[tok] >= vocab_size:
                break
            merges.append(m)
        vocab = {t: i for t, i in self.vocab.items() if i < vocab_size}
        return BPETokenizer(vocab, merges, self.byte_level, self.specials)

    # ==================================================================
    # inference
    # ==================================================================
    def _merge_symbols(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        """Apply merges in rank order to one pre-token's symbols."""
        syms = list(symbols)
        while len(syms) > 1:
            best, best_rank = None, _INF
            for p in zip(syms, syms[1:]):
                r = self.ranks.get(p, _INF)
                if r < best_rank:
                    best, best_rank = p, r
            if best is None:
                break
            a, b = best
            out: list[str] = []
            i = 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            syms = out
        return tuple(syms)

    def trace_word(self, token: str) -> list[dict]:
        """Step-by-step encoding of ONE pre-token (used for the web animation
        and as ground truth for the JavaScript port)."""
        syms = list(_initial_symbols(token, self.byte_level))
        states = [{"symbols": syms[:], "pair": None, "rank": None}]
        while len(syms) > 1:
            best, best_rank = None, _INF
            for p in zip(syms, syms[1:]):
                r = self.ranks.get(p, _INF)
                if r < best_rank:
                    best, best_rank = p, r
            if best is None:
                break
            a, b = best
            out, i = [], 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            syms = out
            states.append({"symbols": syms[:], "pair": [a, b], "rank": best_rank})
        return states

    def _pretokens(self, text: str) -> list[str]:
        return split_gpt2(text) if self.byte_level else split_words(text)

    def _split_specials(self, text: str, allow_special: bool) -> list[tuple[str, bool]]:
        if not (allow_special and self._special_re):
            return [(text, False)]
        parts = self._special_re.split(text)
        return [(p, i % 2 == 1) for i, p in enumerate(parts) if p]

    def tokenize(self, text: str, allow_special: bool = True) -> list[str]:
        out: list[str] = []
        for chunk, is_special in self._split_specials(text, allow_special):
            if is_special:
                out.append(chunk)
                continue
            for pre in self._pretokens(chunk):
                if pre not in self._cache:
                    self._cache[pre] = self._merge_symbols(_initial_symbols(pre, self.byte_level))
                out.extend(self._cache[pre])
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        unk = self.vocab.get("<unk>")
        ids = []
        for tok in self.tokenize(text, allow_special):
            i = self.vocab.get(tok, unk)
            if i is None:
                raise KeyError(f"token {tok!r} not in vocab and no <unk> defined")
            ids.append(i)
        return ids

    def decode(self, ids: list[int]) -> str:
        toks = [self._inv[i] for i in ids]
        if self.byte_level:
            # specials are emitted verbatim; everything else is bytes
            out, buf = [], ""
            for t in toks:
                if t in self.specials:
                    out.append(symbols_to_text(buf)); buf = ""
                    out.append(t)
                else:
                    buf += t
            out.append(symbols_to_text(buf))
            return "".join(out)
        text = "".join(t if t not in self.specials else f" {t} " for t in toks)
        return re.sub(r"\s+", " ", text.replace(EOW, " ")).strip()

    # ==================================================================
    # (de)serialisation
    # ==================================================================
    def to_dict(self) -> dict:
        return {
            "type": "bpe",
            "byte_level": self.byte_level,
            "specials": self.specials,
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        return cls(d["vocab"], [tuple(m) for m in d["merges"]], d["byte_level"], d.get("specials"))
