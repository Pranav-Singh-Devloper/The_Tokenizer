// Playground: train all four tokenizers on one corpus, compare their output on any text.
import { Trainer, WordModel, stats } from "./core.js";
import { h, $, clear, chip, fmt, pct } from "./util.js";
import { PRESETS, loadSample } from "./state.js";

const FAMILIES = [
  { key: "word", name: "Word-level", blurb: "one token per known word; everything else → <unk>" },
  { key: "bpe-char", name: "BPE (characters)", blurb: "Sennrich et al. 2016: chars + </w>, lossy decode" },
  { key: "bpe-byte", name: "Byte-level BPE", blurb: "GPT-2 style: UTF-8 bytes, lossless, never unknown" },
  { key: "wordpiece", name: "WordPiece", blurb: "BERT style: likelihood-ratio merges, greedy longest match, ## pieces" },
];
const MAX_CHIPS = 500;

export async function initPlayground(el) {
  const models = {};
  const ui = {};

  el.append(
    h("h2", {}, "Playground"),
    h("p.lead", {}, "Pick a training corpus and a vocabulary size, train all four tokenizers right here in your browser, then type anything and compare how each one segments it."),
  );

  // ---- controls
  const corpusSel = h("select", {}, Object.entries({ shakespeare: { label: "Shakespeare excerpt (80 KB)" }, ...PRESETS, custom: { label: "Custom text…" } })
    .map(([k, v]) => h("option", { value: k }, v.label)));
  const vocab = h("input", { type: "range", min: 300, max: 2000, step: 50, value: 600 });
  const vocabLbl = h("b", {}, "600");
  vocab.addEventListener("input", () => (vocabLbl.textContent = vocab.value));
  const custom = h("textarea", { placeholder: "Paste training text here…", hidden: true, rows: 5 });
  corpusSel.addEventListener("change", () => (custom.hidden = corpusSel.value !== "custom"));
  const trainBtn = h("button.btn.primary", { onclick: train }, "Train all four");
  const prog = h("div.meter", { hidden: true, style: { width: "160px" } }, h("i", { style: { width: "0%" } }));
  const status = h("span.small.muted", {}, "Not trained yet.");

  el.append(h("div.card", {},
    h("div.row", {},
      h("label.f", {}, "Training corpus", corpusSel),
      h("label.f", {}, h("span", {}, "Target vocabulary size: ", vocabLbl), vocab),
      h("div", {}, trainBtn), prog, status),
    custom));

  // ---- input
  const input = h("textarea", { rows: 3 });
  input.value = "Shakespeare's unbelievably tokenization-proof sonnets, in 2024 — café naïve 🐱";
  let timer;
  input.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(render, 120); });
  el.append(h("div.card", {}, h("label.f", {}, "Text to tokenize", input)));

  ui.out = h("div.card", {}, h("div.muted", {}, "Training…"));
  el.append(ui.out);
  el.append(h("div.card", {},
    h("h3", {}, "What to look for"),
    h("ul.small", {},
      h("li", {}, "Red chips are ", h("code", {}, "<unk>"), ": information destroyed. Word-level produces them for every unseen word; try inventing one."),
      h("li", {}, "Byte-level BPE is the only one that is always lossless (", h("b", {}, "lossless ✓"), ") — try emoji or another script."),
      h("li", {}, "Raise the vocabulary size: tokens per word (fertility) falls, but the embedding table grows. That trade-off is the Experiments tab."),
      h("li", {}, "WordPiece's pure ratio score compresses poorly at small vocabularies — it favours pairs of ", h("em", {}, "rare"), " symbols. The report discusses this and an alternative score."))));

  async function train() {
    trainBtn.disabled = true; prog.hidden = false;
    const V = +vocab.value;
    const text = corpusSel.value === "shakespeare" ? await loadSample()
      : corpusSel.value === "custom" ? custom.value : PRESETS[corpusSel.value].text;
    if (!text.trim()) { status.textContent = "Corpus is empty."; trainBtn.disabled = false; return; }
    status.textContent = "Training…";
    const t0 = performance.now();
    models.word = WordModel.train(text, V);
    let done = 0;
    for (const mode of ["bpe-char", "bpe-byte", "wordpiece"]) {
      const tr = new Trainer(text, { mode });
      await tr.runAsync(V, (p) => { prog.firstChild.style.width = ((done + p) / 3) * 100 + "%"; });
      models[mode] = tr.model();
      done++;
    }
    prog.firstChild.style.width = "100%";
    status.textContent = `Trained in ${((performance.now() - t0) / 1000).toFixed(1)} s on ${text.length.toLocaleString()} characters.`;
    trainBtn.disabled = false; prog.hidden = true;
    render();
  }

  function render() {
    const text = input.value;
    clear(ui.out);
    if (!models.word) return;
    for (const f of FAMILIES) {
      const m = models[f.key];
      const toks = m.tokenize(text), ids = m.encode(text), s = stats(m, text);
      const chips = h("div.chips", {}, toks.slice(0, MAX_CHIPS).map((t, i) =>
        chip(t, { mode: m.mode, id: ids[i], cls: m.isUnk(t) ? "unk" : "" })));
      if (toks.length > MAX_CHIPS) chips.append(h("span.muted", {}, `… +${toks.length - MAX_CHIPS} more`));
      ui.out.append(h("div.tokrow", {},
        h("header", {},
          h("b", {}, f.name),
          h("span.stat", {}, "vocab ", h("b", {}, m.vocabSize)),
          h("span.stat", {}, h("b", {}, s.tokens), " tokens"),
          h("span.stat", {}, h("b", {}, fmt(s.charsPerToken)), " chars/token"),
          h("span.stat", {}, h("b", {}, fmt(s.fertility)), " tokens/word"),
          s.unk ? h("span.pill.bad", {}, `${s.unk} unknown (${pct(s.unkRate)})`) : h("span.pill.ok", {}, "no unknowns"),
          s.lossless ? h("span.pill.ok", {}, "lossless ✓") : h("span.pill.warn", {}, "lossy decode")),
        h("div.small.muted", {}, f.blurb),
        h("div", { style: { marginTop: "8px" } }, chips)));
    }
  }

  await train();
}
