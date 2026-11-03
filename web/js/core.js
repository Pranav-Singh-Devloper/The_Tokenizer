/**
 * core.js - the tokenizer algorithms, ported 1:1 from the Python library
 * (src/tokenlab) so the browser demo runs the *real* algorithm, not a mock-up.
 *
 * Parity with Python is enforced by web/tests/parity.test.mjs against golden
 * fixtures exported from the Python implementation (`npm test`).
 *
 * No dependencies, no build step: plain ES modules.
 */

// ---------------------------------------------------------------- strings
/** Compare two strings by Unicode code point (Python's ordering), not UTF-16 unit. */
export function cmpStr(a, b) {
  const x = Array.from(a), y = Array.from(b);
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i++) {
    const d = x[i].codePointAt(0) - y[i].codePointAt(0);
    if (d !== 0) return d;
  }
  return x.length - y.length;
}
export const cmpPair = (p, q) => cmpStr(p[0], q[0]) || cmpStr(p[1], q[1]);

// ------------------------------------------------------- pre-tokenization
// stdlib-`re` equivalents of \p{L}/\p{N}, mirrored here so both sides agree:
//   letter = L + Nl + No   digit = Nd   word char = L + N + "_"
const L = "\\p{L}\\p{Nl}\\p{No}";
const GPT2_RE = new RegExp(
  `'s|'t|'re|'ve|'m|'ll|'d| ?[${L}]+| ?\\p{Nd}+| ?(?:[^\\s\\p{L}\\p{N}_]|_)+|\\s+(?!\\S)|\\s+`, "gu");
const WORD_RE = /[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu;

export const splitGpt2 = (t) => t.match(GPT2_RE) ?? [];
export const splitWords = (t) => t.match(WORD_RE) ?? [];
export const stripAccents = (t) => t.normalize("NFD").replace(/\p{Mn}/gu, "");
export const splitBasic = (t, lower = true) =>
  splitWords(lower ? stripAccents(t.toLowerCase()) : t);

// ------------------------------------------------------ byte <-> unicode
let _b2u = null, _u2b = null;
export function bytesToUnicode() {
  if (_b2u) return _b2u;
  const printable = [];
  const range = (a, b) => { for (let i = a; i <= b; i++) printable.push(i); };
  range(33, 126); range(161, 172); range(174, 255);
  const have = new Set(printable);
  const bs = printable.slice(), cs = printable.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!have.has(b)) { bs.push(b); cs.push(256 + n++); }
  _b2u = new Map(bs.map((b, i) => [b, String.fromCodePoint(cs[i])]));
  _u2b = new Map(bs.map((b, i) => [String.fromCodePoint(cs[i]), b]));
  return _b2u;
}
export const textToSymbols = (text) => {
  const m = bytesToUnicode();
  return Array.from(new TextEncoder().encode(text), (b) => m.get(b));
};
export const symbolsToText = (s) => {
  bytesToUnicode();
  return new TextDecoder().decode(Uint8Array.from(Array.from(s), (c) => _u2b.get(c)));
};

// ----------------------------------------------------------------- shared
export const EOW = "</w>", CONT = "##";
export const CLASSIC_SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"];
export const BYTE_SPECIALS = ["<|endoftext|>"];
export const WP_SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];
const INF = Number.MAX_SAFE_INTEGER;
const pairKey = (a, b) => a + "\u0001" + b;

function initialSymbols(token, mode) {
  if (mode === "bpe-byte") return textToSymbols(token);
  if (mode === "bpe-char") {
    const c = Array.from(token);
    c[c.length - 1] += EOW;
    return c;
  }
  return Array.from(token).map((c, i) => (i === 0 ? c : CONT + c)); // wordpiece
}
const pretokenizer = (mode, lower = true) =>
  mode === "bpe-byte" ? splitGpt2 : mode === "bpe-char" ? splitWords : (t) => splitBasic(t, lower);

export const mergedSymbol = (a, b) => a + (b.startsWith(CONT) ? b.slice(CONT.length) : b);

// ================================================================ TRAINER
/**
 * Step-able trainer for BPE (char / byte level) and WordPiece.
 *
 * `analyze()` counts pairs and chooses the winner (phase 1+2),
 * `commit()` rewrites the corpus (phase 3). `step()` does both.
 * The web visualizer drives these phases separately to animate them.
 *
 * Deliberately naive (recount all pairs every step): O(unique words) per merge,
 * which keeps the code identical to the textbook description. The Python
 * library has the incremental version (merge_engine.py).
 */
export class Trainer {
  constructor(text, { mode = "bpe-byte", lowercase = true } = {}) {
    this.mode = mode;
    this.specials = mode === "bpe-byte" ? BYTE_SPECIALS : mode === "bpe-char" ? CLASSIC_SPECIALS : WP_SPECIALS;
    const counts = new Map();
    for (const w of pretokenizer(mode, lowercase)(text)) counts.set(w, (counts.get(w) ?? 0) + 1);
    this.words = [...counts].map(([text, freq]) => ({ text, freq, symbols: initialSymbols(text, mode) }));
    const base = mode === "bpe-byte"
      ? Array.from({ length: 256 }, (_, b) => bytesToUnicode().get(b))
      : [...new Set(this.words.flatMap((w) => w.symbols))].sort(cmpStr);
    this.vocab = new Map();
    for (const t of [...this.specials, ...base]) if (!this.vocab.has(t)) this.vocab.set(t, this.vocab.size);
    this.baseSize = this.vocab.size;
    this.merges = [];       // [[a, b], ...]
    this.history = [];      // one record per committed merge
    this.pending = null;
  }

  /** Phase 1+2: count pairs, score them, choose the winner. */
  analyze() {
    const pairs = new Map(), syms = new Map();
    for (const w of this.words) {
      for (const s of w.symbols) syms.set(s, (syms.get(s) ?? 0) + w.freq);
      for (let i = 0; i + 1 < w.symbols.length; i++) {
        const a = w.symbols[i], b = w.symbols[i + 1], k = pairKey(a, b);
        const e = pairs.get(k);
        if (e) e.count += w.freq; else pairs.set(k, { pair: [a, b], count: w.freq });
      }
    }
    if (pairs.size === 0) return (this.pending = null);
    const wp = this.mode === "wordpiece";
    const rows = [...pairs.values()].map((e) => {
      const ca = syms.get(e.pair[0]), cb = syms.get(e.pair[1]);
      return { ...e, ca, cb, den: ca * cb, score: wp ? e.count / (ca * cb) : e.count };
    });
    // BPE: max count. WordPiece: max count/(ca*cb) via exact cross-multiplication.
    // Ties: smallest pair (code-point order). Identical to the Python rules.
    const better = (x, y) => {  // is x strictly better than y?
      if (wp) {
        const l = x.count * y.den, r = y.count * x.den;
        if (l !== r) return l > r;
      } else if (x.count !== y.count) return x.count > y.count;
      return cmpPair(x.pair, y.pair) < 0;
    };
    let best = rows[0];
    for (const r of rows) if (better(r, best)) best = r;
    rows.sort((x, y) => (x === y ? 0 : better(x, y) ? -1 : 1));
    const [a, b] = best.pair;
    const affected = this.words.filter((w) => w.symbols.some((s, i) => s === a && w.symbols[i + 1] === b));
    this.pending = {
      step: this.merges.length + 1,
      ranking: rows, best, newSymbol: mergedSymbol(a, b),
      totalPairs: rows.length, affected: affected.map((w) => w.text),
    };
    return this.pending;
  }

  /** Phase 3: apply the pending merge to every word. */
  commit() {
    const p = this.pending ?? this.analyze();
    if (!p) return null;
    const [a, b] = p.best.pair;
    for (const w of this.words) {
      const out = [];
      for (let i = 0; i < w.symbols.length; ) {
        if (i + 1 < w.symbols.length && w.symbols[i] === a && w.symbols[i + 1] === b) {
          out.push(p.newSymbol); i += 2;
        } else out.push(w.symbols[i++]);
      }
      w.symbols = out;
    }
    this.merges.push([a, b]);
    const isNew = !this.vocab.has(p.newSymbol);
    if (isNew) this.vocab.set(p.newSymbol, this.vocab.size);
    const rec = { ...p, isNew, vocabSize: this.vocab.size };
    this.history.push(rec);
    this.pending = null;
    return rec;
  }

  step() { return this.analyze() && this.commit(); }

  run(untilVocab) {
    while (this.vocab.size < untilVocab && this.step()) { /* keep merging */ }
    return this;
  }

  /** Async variant that yields to the event loop so the UI stays responsive. */
  async runAsync(untilVocab, onProgress) {
    let n = 0;
    while (this.vocab.size < untilVocab && this.step()) {
      if (++n % 25 === 0) { onProgress?.(this.vocab.size / untilVocab); await new Promise((r) => setTimeout(r)); }
    }
    return this;
  }

  /** Freeze into an inference model. */
  model() {
    if (this.mode === "wordpiece") return new WordPieceModel(new Map(this.vocab));
    return new BPEModel(new Map(this.vocab), this.merges.map((m) => m.slice()), this.mode === "bpe-byte");
  }
}

// ==================================================================== BPE
export class BPEModel {
  constructor(vocab, merges, byteLevel) {
    this.vocab = vocab; this.merges = merges; this.byteLevel = byteLevel;
    this.mode = byteLevel ? "bpe-byte" : "bpe-char";
    this.ranks = new Map(merges.map(([a, b], i) => [pairKey(a, b), i]));
    this.inv = new Map([...vocab].map(([t, i]) => [i, t]));
    this.unk = vocab.get("<unk>");
    this.cache = new Map();
  }
  get vocabSize() { return this.vocab.size; }

  /** One-pre-token encoding with a full trace (ground truth for the animation). */
  trace(token) {
    let syms = initialSymbols(token, this.mode);
    const states = [{ symbols: syms.slice(), pair: null, rank: null, candidates: this._cands(syms) }];
    for (;;) {
      let best = null, bestRank = INF;
      for (let i = 0; i + 1 < syms.length; i++) {
        const r = this.ranks.get(pairKey(syms[i], syms[i + 1])) ?? INF;
        if (r < bestRank) { best = [syms[i], syms[i + 1]]; bestRank = r; }
      }
      if (!best) break;
      const out = [];
      for (let i = 0; i < syms.length; ) {
        if (i + 1 < syms.length && syms[i] === best[0] && syms[i + 1] === best[1]) { out.push(best[0] + best[1]); i += 2; }
        else out.push(syms[i++]);
      }
      syms = out;
      states.push({ symbols: syms.slice(), pair: best, rank: bestRank, candidates: this._cands(syms) });
    }
    return states;
  }
  _cands(syms) {
    return syms.slice(0, -1).map((s, i) => ({
      i, pair: [s, syms[i + 1]], rank: this.ranks.get(pairKey(s, syms[i + 1])) ?? null,
    }));
  }
  pretokens(text) { return (this.byteLevel ? splitGpt2 : splitWords)(text); }
  tokenizeWord(pre) {
    let t = this.cache.get(pre);
    if (!t) { const st = this.trace(pre); t = st[st.length - 1].symbols; this.cache.set(pre, t); }
    return t;
  }
  tokenize(text) { return this.pretokens(text).flatMap((p) => this.tokenizeWord(p)); }
  encode(text) { return this.tokenize(text).map((t) => this.vocab.get(t) ?? this.unk); }
  /** token string -> display string (bytes shown as text where possible) */
  decode(ids) {
    const toks = ids.map((i) => this.inv.get(i));
    if (this.byteLevel) return symbolsToText(toks.join(""));
    return toks.join("").replaceAll(EOW, " ").replace(/\s+/g, " ").trim();
  }
  isUnk(tok) { return !this.vocab.has(tok); }
}

// ============================================================== WordPiece
export class WordPieceModel {
  constructor(vocab, lowercase = true) {
    this.vocab = vocab; this.lowercase = lowercase; this.mode = "wordpiece";
    this.inv = new Map([...vocab].map(([t, i]) => [i, t]));
    this.unk = vocab.get("[UNK]");
  }
  get vocabSize() { return this.vocab.size; }

  /** Greedy longest-match-first with a log of every candidate tried. */
  traceWord(word) {
    const chars = Array.from(word), log = [], pieces = [];
    if (chars.length > 100) return { pieces: ["[UNK]"], log, failed: true };
    let start = 0;
    while (start < chars.length) {
      let end = chars.length, found = null;
      while (start < end) {
        let piece = chars.slice(start, end).join("");
        if (start > 0) piece = CONT + piece;
        const hit = this.vocab.has(piece);
        log.push({ start, end, piece, hit });
        if (hit) { found = piece; break; }
        end--;
      }
      if (found === null) return { pieces: ["[UNK]"], log, failed: true };
      pieces.push(found);
      start = end;
    }
    return { pieces, log, failed: false };
  }
  pretokens(text) { return splitBasic(text, this.lowercase); }
  tokenize(text) { return this.pretokens(text).flatMap((w) => this.traceWord(w).pieces); }
  encode(text) { return this.tokenize(text).map((t) => this.vocab.get(t) ?? this.unk); }
  decode(ids) {
    const words = [];
    for (const i of ids) {
      const t = this.inv.get(i);
      if (t.startsWith(CONT) && words.length) words[words.length - 1] += t.slice(CONT.length);
      else words.push(t);
    }
    return words.join(" ");
  }
  isUnk(tok) { return tok === "[UNK]"; }
}

// ================================================================== WORD
export class WordModel {
  constructor(vocab, lowercase = true) {
    this.vocab = vocab; this.lowercase = lowercase; this.mode = "word";
    this.inv = new Map([...vocab].map(([t, i]) => [i, t]));
  }
  static train(text, maxSize = 10000, { lowercase = true, minFreq = 1 } = {}) {
    const counts = new Map();
    for (const w of splitBasic(text, lowercase)) counts.set(w, (counts.get(w) ?? 0) + 1);
    const ranked = [...counts].filter(([, c]) => c >= minFreq)
      .sort((a, b) => b[1] - a[1] || cmpStr(a[0], b[0])).map(([w]) => w);
    const vocab = new Map();
    for (const t of [...CLASSIC_SPECIALS, ...ranked.slice(0, Math.max(maxSize - 4, 0))]) vocab.set(t, vocab.size);
    return new WordModel(vocab, lowercase);
  }
  get vocabSize() { return this.vocab.size; }
  pretokens(text) { return splitBasic(text, this.lowercase); }
  tokenize(text) { return this.pretokens(text).map((w) => (this.vocab.has(w) ? w : "<unk>")); }
  encode(text) { return this.tokenize(text).map((t) => this.vocab.get(t)); }
  decode(ids) { return ids.map((i) => this.inv.get(i)).join(" "); }
  isUnk(tok) { return tok === "<unk>"; }
}

// ============================================================== metrics
export function stats(model, text) {
  const toks = model.tokenize(text);
  const words = Math.max(splitWords(text).length, 1);
  const unk = toks.filter((t) => model.isUnk(t)).length;
  return {
    tokens: toks.length, chars: Array.from(text).length,
    bytes: new TextEncoder().encode(text).length,
    charsPerToken: Array.from(text).length / Math.max(toks.length, 1),
    fertility: toks.length / words, unk, unkRate: unk / Math.max(toks.length, 1),
    lossless: model.decode(model.encode(text)) === text,
  };
}
