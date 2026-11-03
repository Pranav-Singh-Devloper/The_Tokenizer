// Experiments tab: renders results/summary.json produced by experiments/run_experiments.py.
import { h, clear, fmt, pct, chip } from "./util.js";

const COLORS = { word: "#d95f02", "char-BPE": "#1b9e77", "byte-BPE": "#7570b3", WordPiece: "#e7298a", "WordPiece-gain": "#66a61e" };
const METRICS = {
  chars_per_token: { label: "Compression (characters per token)", better: "higher is better", f: (v) => fmt(v) },
  fertility: { label: "Fertility (tokens per word)", better: "lower is better", f: (v) => fmt(v) },
  unk_rate: { label: "Unknown-token rate", better: "lower is better", f: (v) => pct(v), scale: 100 },
};
const NS = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs = {}, ...kids) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  kids.forEach((k) => e.append(k));
  return e;
};

/** Minimal multi-series line chart. series: [{name,color,points:[[x,y],...]}] */
function lineChart(series, { xlog = true, logy = false, xlabel = "", yfmt = (v) => v } = {}) {
  const W = 640, H = 340, m = { l: 52, r: 14, t: 12, b: 40 };
  const pts = series.flatMap((s) => s.points);
  const tx = (x) => (xlog ? Math.log10(x) : x), ty = (y) => (logy ? Math.log10(y) : y);
  const xs = pts.map((p) => tx(p[0])), ys = pts.map((p) => ty(p[1]));
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (!logy) { const pad = (y1 - y0) * 0.08 || 1; y0 = Math.min(y0, y0 - pad); y1 += pad; if (y0 > 0 && y0 < pad * 4) y0 = 0; }
  const X = (x) => m.l + ((tx(x) - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (y) => H - m.b - ((ty(y) - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img" });
  const grid = svgEl("g", { class: "grid" });
  for (let i = 0; i <= 5; i++) {
    const yv = y0 + ((y1 - y0) * i) / 5, y = H - m.b - (i / 5) * (H - m.t - m.b);
    grid.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: y, y2: y }));
    const t = svgEl("text", { x: m.l - 6, y: y + 4, "text-anchor": "end" }); t.textContent = yfmt(logy ? 10 ** yv : yv);
    grid.append(t);
  }
  const xt = [...new Set(pts.map((p) => p[0]))];
  xt.forEach((xv) => {
    const t = svgEl("text", { x: X(xv), y: H - m.b + 16, "text-anchor": "middle" }); t.textContent = xv >= 1000 ? xv / 1000 + "k" : xv;
    grid.append(t);
  });
  const xl = svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }); xl.textContent = xlabel;
  svg.append(grid, xl);
  for (const s of series) {
    svg.append(svgEl("polyline", { points: s.points.map((p) => `${X(p[0])},${Y(p[1])}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 2.2 }));
    for (const p of s.points) {
      const c = svgEl("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: s.color });
      c.append(svgEl("title", {}, document.createTextNode(`${s.name}: V=${p[0]} → ${yfmt(p[1])}`)));
      svg.append(c);
    }
  }
  return svg;
}

export async function initExperiments(el) {
  el.append(h("h2", {}, "Experiments"), h("p.lead", {}, "Loading results…"));
  let d;
  try { d = await (await fetch("data/experiments.json")).json(); }
  catch { clear(el).append(h("h2", {}, "Experiments"), h("p", {}, "Run ", h("code", {}, "python experiments/run_experiments.py"), " to generate results.")); return; }
  clear(el);

  const c = d.corpus;
  el.append(
    h("h2", {}, "Experiments"),
    h("p.lead", {}, `Every tokenizer was trained on ${(c.train_chars / 1e6).toFixed(2)} M characters of Tiny Shakespeare and evaluated on a held-out ${(c.test_chars / 1e3).toFixed(0)} K characters it never saw. `
      + `${(c.test_oov_word_rate * 100).toFixed(1)}% of the test words never occur in the training text — that is the open-vocabulary problem in one number.`));

  // --- E1
  const metric = h("select", {}, Object.entries(METRICS).map(([k, v]) => h("option", { value: k }, v.label)));
  const off = new Set();
  const chartBox = h("div"), legend = h("div.legend");
  const draw = () => {
    const M = METRICS[metric.value];
    const names = Object.keys(COLORS);
    clear(legend).append(...names.map((n) => h("button", { class: off.has(n) ? "off" : "", onclick: () => { off.has(n) ? off.delete(n) : off.add(n); draw(); } },
      h("i", { style: { background: COLORS[n] } }), n)));
    const series = names.filter((n) => !off.has(n)).map((n) => ({
      name: n, color: COLORS[n],
      points: d.e1.filter((r) => r.tokenizer === n).map((r) => [r.target_vocab, r[metric.value] * (M.scale ?? 1)]),
    }));
    clear(chartBox).append(lineChart(series, { xlabel: "vocabulary size (log scale)", yfmt: (v) => (M.scale ? v.toFixed(0) + "%" : v.toFixed(1)) }));
  };
  metric.addEventListener("change", draw);
  el.append(h("div.card", {},
    h("h3", {}, "E1 · Vocabulary-size sweep"),
    h("p.small.muted", {}, "One training run per family; smaller vocabularies are prefixes of the same merge list (BPE/WordPiece vocabularies are nested)."),
    h("label.f", {}, "Metric", metric), legend, chartBox,
    h("p.small", {}, h("b", {}, "Reading it: "), "the word-level model has fertility 1 by construction and the best-looking ‘compression’, but ",
      h("b", {}, "11% of its tokens are "), h("code", {}, "<unk>"), " at V=2000 and still ~4% at V=10 832 (every training word). Sub-word models never lose information; they pay in sequence length instead.")));
  draw();

  // --- E2
  el.append(h("div.card", {}, h("h3", {}, `E2 · Head-to-head at V = 2000`),
    h("div.scroll", {}, h("table", {},
      h("thead", {}, h("tr", {}, ...["Tokenizer", "Train time (s)", "chars/token", "tokens/word", "unk rate", "lossless"].map((t, i) => h("th", { class: i ? "num" : "" }, t)))),
      h("tbody", {}, d.e2.map((r) => h("tr", {}, h("td", {}, r.tokenizer), h("td.num", {}, fmt(r.train_seconds)), h("td.num", {}, fmt(r.chars_per_token)),
        h("td.num", {}, fmt(r.fertility)), h("td.num", {}, pct(r.unk_rate)), h("td.num", {}, r.roundtrip_exact ? "✓" : "✗")))))),
    h("p.small.muted", {}, "‘Lossless’ means decode(encode(x)) == x exactly. Only byte-level BPE keeps case, whitespace and every character. Note chars/token is not strictly like-for-like: the lossy pipelines discard whitespace instead of spending tokens on it.")));

  // --- E3
  const e3 = d.e3, fams = Object.keys(Object.values(e3)[0]);
  el.append(h("div.card", {}, h("h3", {}, "E3 · How unseen words are split (V = 2000)"),
    h("div.scroll", {}, h("table.seg-table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "Unseen word"), ...fams.map((f) => h("th", {}, f)))),
      h("tbody", {}, Object.entries(e3).map(([w, m]) => h("tr", {}, h("td", {}, w), ...fams.map((f) =>
        h("td", {}, h("div.chips", {}, m[f].map((t) => chip2(t, f)))))))))),
    h("p.small.muted", {}, "Plain WordPiece (ratio score) falls back to single characters; the frequency-weighted variant finds morphemes (counter·re·vol·ution·ary).")));

  // --- E4
  const doms = [...new Set(d.e4.map((r) => r.domain))], toks = [...new Set(d.e4.map((r) => r.tokenizer))];
  el.append(h("div.card", {}, h("h3", {}, "E4 · Domain shift (bytes per token — higher = better compression)"),
    h("div.scroll", {}, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "Tokenizer"), ...doms.map((x) => h("th.num", {}, x)))),
      h("tbody", {}, toks.map((t) => h("tr", {}, h("td", {}, t), ...doms.map((dm) => {
        const r = d.e4.find((x) => x.domain === dm && x.tokenizer === t);
        return h("td.num", {}, r ? r.bytes_per_token.toFixed(2) + (r.unk_rate > 0.001 ? ` (${pct(r.unk_rate)} unk)` : "") : "–");
      })))))),
    h("p.small.muted", {}, "A Shakespeare-trained 2k vocabulary degrades on modern prose and collapses on code, while GPT-2's 50k vocabulary (trained on web text) holds up — the tokenizer encodes its training distribution.")));

  // --- E5 Zipf
  const z = d.zipf;
  el.append(h("div.card", {}, h("h3", {}, "E5 · Token frequency vs. rank (Zipf)"),
    lineChartZ(z),
    h("p.small.muted", {}, "Word frequencies fall off as a long power-law tail (many words seen once or twice). BPE's merge procedure flattens the head and thins the tail: the rare-token problem is far less severe.")));
}

function lineChartZ(z) {
  const series = Object.entries(z).map(([name, ys]) => ({
    name, color: name === "word" ? COLORS.word : COLORS["byte-BPE"],
    points: ys.map((y, i) => [i + 1, y]).filter(([r]) => r <= 30 || r % 25 === 0),
  }));
  const box = h("div");
  box.append(lineChart(series, { xlog: true, logy: true, xlabel: "rank (log)", yfmt: (v) => (v >= 1000 ? Math.round(v / 1000) + "k" : Math.round(v)) }),
    h("div.legend", {}, h("span", {}, h("i", { style: { background: COLORS.word, display: "inline-block", width: "14px", height: "4px" } }), " word"),
      h("span", { style: { marginLeft: "14px" } }, h("i", { style: { background: COLORS["byte-BPE"], display: "inline-block", width: "14px", height: "4px" } }), " byte-BPE (V=2000)")));
  return box;
}

const chip2 = (t, fam) => chip(t, { mode: fam === "byte-BPE" ? "bpe-byte" : fam === "char-BPE" ? "bpe-char" : fam.startsWith("Word") && fam !== "word" ? "wordpiece" : "word", cls: /^(<unk>|\[UNK\])$/.test(t) ? "unk" : "" });
