// Training lab: the step-by-step simulation of BPE / WordPiece training.
//
// Each merge is shown as three phases:
//   1 COUNT  - count every adjacent pair (weighted by word frequency)
//   2 CHOOSE - pick the winner by the algorithm's scoring rule
//   3 MERGE  - rewrite the corpus and add the new symbol to the vocabulary
import { Trainer } from "./core.js";
import { h, $, clear, chip, display, fmt, sleep, sparkline } from "./util.js";
import { PRESETS, shared } from "./state.js";

const ALGOS = {
  "bpe-char": "BPE · characters (Sennrich 2016)",
  "bpe-byte": "Byte-level BPE (GPT-2)",
  wordpiece: "WordPiece (BERT)",
};
const MAX_WORDS_SHOWN = 40;

export function initTrainLab(el) {
  let trainer, phase = 0, lastMerge = null, playing = false, exhausted = false;

  // ---------------------------------------------------------------- controls
  const algo = h("select", {}, Object.entries(ALGOS).map(([k, v]) => h("option", { value: k }, v)));
  const preset = h("select", {}, [...Object.entries(PRESETS).map(([k, v]) => h("option", { value: k }, v.label)),
    h("option", { value: "custom" }, "Custom (edit below)")]);
  const merges = h("input", { type: "number", min: 1, max: 400, value: 12, style: { width: "80px" } });
  const corpus = h("textarea", { rows: 3, spellcheck: false });
  const speed = h("input", { type: "range", min: 1, max: 10, value: 5, "aria-label": "Playback speed" });
  const btn = {
    reset: h("button.btn", { onclick: () => reset() }, "⟲ Reset"),
    phase: h("button.btn", { onclick: () => advance() }, "Next phase ›"),
    merge: h("button.btn", { onclick: () => nextMerge() }, "Next merge ››"),
    play: h("button.btn.primary", { onclick: togglePlay }, "▶ Play"),
    all: h("button.btn", { onclick: runAll }, "Run all ⏭"),
  };
  corpus.value = PRESETS.toy.text;
  preset.addEventListener("change", () => { if (preset.value !== "custom") { corpus.value = PRESETS[preset.value].text; reset(); } });
  corpus.addEventListener("input", () => (preset.value = "custom"));
  algo.addEventListener("change", () => reset());
  merges.addEventListener("change", render);

  el.append(
    h("h2", {}, "Training lab"),
    h("p.lead", {}, "Watch the merge loop that builds a sub-word vocabulary. Every merge has three phases — ", h("b", {}, "count"), " the pairs, ",
      h("b", {}, "choose"), " the winner, ", h("b", {}, "merge"), " it everywhere. The numbers on screen are the real algorithm's, not an illustration."),
    h("div.card.lab-controls", {},
      h("div.row", {},
        h("label.f", {}, "Algorithm", algo), h("label.f", {}, "Corpus", preset),
        h("label.f", {}, "Merges to run", merges), h("div", { style: { alignSelf: "end" } }, btn.reset)),
      h("label.f", {}, "Training text (pre-tokenised into words; frequencies matter)", corpus),
      h("div.row", {}, btn.phase, btn.merge, btn.play,
        h("label.f", { style: { flexDirection: "row", alignItems: "center", gap: "8px" } }, "Speed", speed), h("span.sep"), btn.all)));

  const ui = {
    phases: h("div.phases"), narr: h("div.narration"),
    corpus: h("div"), pairs: h("div"), stats: h("div"), merges: h("div.mergelist"), spark: h("div"),
  };
  el.append(
    h("div.card", {}, ui.phases, ui.narr),
    h("div.grid.lab", {},
      h("div.card", {}, h("h3", {}, "Corpus ", h("span.small.muted", {}, "(frequency × current symbols)")), ui.corpus),
      h("div.card", {}, h("h3", {}, "Pair statistics"), ui.pairs),
      h("div.card", {}, h("h3", {}, "Vocabulary"), ui.stats, h("div.small.muted", { style: { margin: "10px 0 2px" } }, "corpus length in symbols, per merge"), ui.spark,
        h("h3", { style: { marginTop: "14px" } }, "Merge rules (newest first)"), ui.merges)));

  // ------------------------------------------------------------------- logic
  const limit = () => Math.max(1, +merges.value || 1);
  const canAdvance = () => !exhausted && (phase !== 0 || trainer.merges.length < limit());

  function reset() {
    playing = false;
    trainer = new Trainer(corpus.value, { mode: algo.value });
    phase = 0; lastMerge = null; exhausted = false;
    shared.trainer = trainer;
    render();
  }

  function advance() {
    if (!canAdvance()) return false;
    if (phase === 0) {
      lastMerge = null;
      if (!trainer.analyze()) { exhausted = true; render(); return false; }
      phase = 1;
    } else if (phase === 1) phase = 2;
    else { lastMerge = trainer.commit(); phase = 0; }
    render();
    return true;
  }
  function nextMerge() { do { if (!advance()) break; } while (phase !== 0); }
  async function togglePlay() {
    if (playing) { playing = false; render(); return; }
    playing = true; render();
    while (playing && advance()) await sleep(1700 - speed.value * 160);
    playing = false; render();
  }
  async function runAll() {
    playing = false; phase = 0; trainer.pending = null;
    let n = 0;
    while (trainer.merges.length < limit()) {
      if (!trainer.step()) { exhausted = true; break; }
      if (++n % 20 === 0) await sleep(0); // keep the page responsive
    }
    lastMerge = trainer.history.at(-1) ?? null;
    render();
  }

  // ------------------------------------------------------------------ render
  const sym = (t, cls = "") => chip(t, { mode: trainer.mode, cls });
  const pairLabel = (p) => `${display(p[0], trainer.mode).text} + ${display(p[1], trainer.mode).text}`;

  function render() {
    const wp = trainer.mode === "wordpiece";
    const pend = phase >= 1 ? trainer.pending : null;
    const done = trainer.merges.length >= limit() && phase === 0;

    // phase indicator
    clear(ui.phases).append(...["1 · Count pairs", "2 · Choose winner", "3 · Merge"].map((t, i) =>
      h("span", { class: (phase === i + 1 || (phase === 0 && lastMerge && i === 2)) ? "on" : (phase > i + 1 ? "done" : "") }, t)));

    // narration
    clear(ui.narr);
    const n = trainer.merges.length + 1;
    if (exhausted) ui.narr.append("No adjacent pairs left: every word is a single symbol. Nothing more to learn.");
    else if (phase === 1) {
      ui.narr.append(h("b.ph", {}, `Merge #${n} · Phase 1 — Count. `),
        `Slide a window of two symbols across every word and add the word's frequency to that pair's tally. ${pend.totalPairs} distinct adjacent pairs found; the ${Math.min(14, pend.totalPairs)} best are shown.`,
        wp ? " WordPiece also needs how often each symbol occurs on its own, to compute the score." : "");
    } else if (phase === 2) {
      const b = pend.best;
      ui.narr.append(h("b.ph", {}, `Merge #${n} · Phase 2 — Choose. `),
        wp ? h("span", {}, "Score each pair with ", h("code", {}, "count(ab) ÷ (count(a) × count(b))"), `. Winner: `, h("b", {}, pairLabel(b.pair)),
          h("code", {}, ` = ${b.count} ÷ (${b.ca} × ${b.cb}) = ${b.score.toPrecision(3)}`), ". Rare symbols that mostly appear together score highest — not necessarily the most frequent pair.")
          : h("span", {}, "The most frequent pair wins: ", h("b", {}, pairLabel(b.pair)), ` occurs ${b.count}× in the corpus. Ties go to the lexicographically smallest pair so training is deterministic.`),
        ` It appears in ${pend.affected.length} distinct word${pend.affected.length === 1 ? "" : "s"} (highlighted).`);
    } else if (lastMerge) {
      ui.narr.append(h("b.ph", {}, `Merge #${lastMerge.step} · Phase 3 — Merge. `),
        `Every occurrence of `, h("b", {}, pairLabel(lastMerge.best.pair)), ` was replaced by the single symbol `, sym(lastMerge.newSymbol, "fresh"),
        lastMerge.isNew ? ` and added to the vocabulary as id ${trainer.vocab.get(lastMerge.newSymbol)}.` : " (this string already existed in the vocabulary).",
        done ? h("span.done-note", {}, "  ✔ Reached the merge budget — raise “Merges to run” to continue.") : "");
    } else {
      ui.narr.append(`Start from ${trainer.baseSize} base symbols (${trainer.specials.length} special + ${trainer.baseSize - trainer.specials.length} ${trainer.mode === "bpe-byte" ? "bytes" : "characters"}). `,
        `Press ▶ Play, or step manually. Each merge adds one symbol; stop after ${limit()} merge${limit() === 1 ? "" : "s"}.`);
    }

    // corpus
    clear(ui.corpus);
    const words = [...trainer.words].sort((a, b) => b.freq - a.freq || (a.text < b.text ? -1 : 1));
    const affected = new Set(pend?.affected ?? []);
    for (const w of words.slice(0, MAX_WORDS_SHOWN)) {
      const [pa, pb] = pend ? pend.best.pair : [null, null];
      const chips = w.symbols.map((s, i) => {
        let cls = "";
        if (pend && phase === 2 && s === pa && w.symbols[i + 1] === pb) cls = "pair-a";
        else if (pend && phase === 2 && i > 0 && w.symbols[i - 1] === pa && s === pb) cls = "pair-b";
        else if (!pend && lastMerge && s === lastMerge.newSymbol) cls = "fresh";
        return sym(s, cls);
      });
      ui.corpus.append(h("div.corpus-row", { class: phase === 2 && affected.has(w.text) ? "affected" : "" },
        h("div.freq", {}, `${w.freq}×`), h("div.chips", {}, chips)));
    }
    if (words.length > MAX_WORDS_SHOWN) ui.corpus.append(h("div.small.muted", { style: { paddingTop: "8px" } }, `… and ${words.length - MAX_WORDS_SHOWN} more distinct words (all are used in training).`));

    // pair table
    clear(ui.pairs);
    if (!pend) ui.pairs.append(h("div.muted.small", {}, phase === 0 && lastMerge ? "Pair counts are recomputed at the start of the next merge (phase 1)." : "Press “Next phase” to count pairs."));
    else {
      const val = (r) => (wp ? r.score : r.count);
      const max = Math.max(...pend.ranking.map(val));
      for (const r of pend.ranking) {
        const win = phase === 2 && r === pend.best;
        ui.pairs.append(h("div.pairrow", { class: (win ? "win " : "") + (wp ? "wp" : ""), title: wp ? `count=${r.count}, count(a)=${r.ca}, count(b)=${r.cb}` : "" },
          h("span.pair", {}, pairLabel(r.pair)),
          h("div", {}, h("div.bar", { style: { width: Math.max(2, (val(r) / max) * 100) + "%" } })),
          h("span.val", {}, wp ? r.score.toPrecision(3) : r.count)));
      }
      ui.pairs.append(h("div.small.muted", { style: { paddingTop: "8px" } }, wp ? "bars: count(ab) ÷ (count(a)·count(b)) — hover for the parts" : "bars: pair frequency in the corpus"));
    }

    // vocabulary stats
    const len = trainer.corpusLen();
    clear(ui.stats).append(
      h("div.bigstat", {},
        h("div", {}, h("b", {}, trainer.vocab.size), h("span", {}, "vocab size")),
        h("div", {}, h("b", {}, `${trainer.merges.length}/${limit()}`), h("span", {}, "merges")),
        h("div", {}, h("b", {}, len), h("span", {}, "corpus symbols")),
        h("div", {}, h("b", {}, fmt(trainer.initialLen / len) + "×"), h("span", {}, "compression"))),
      h("div.meter", { style: { marginTop: "10px" } }, h("i", { style: { width: Math.min(100, (trainer.merges.length / limit()) * 100) + "%" } })));
    clear(ui.spark).append(sparkline([trainer.initialLen, ...trainer.history.map((r) => r.corpusLen)], { label: "corpus length per merge" }));

    // merge list
    clear(ui.merges);
    trainer.history.slice().reverse().slice(0, 200).forEach((r) => ui.merges.append(h("div", {},
      h("span.n", {}, `#${r.step}`), sym(r.best.pair[0]), "+", sym(r.best.pair[1]), "→", sym(r.newSymbol),
      h("span.muted", {}, wp ? `s=${r.best.score.toPrecision(2)}` : `×${r.best.count}`))));
    if (!trainer.history.length) ui.merges.append(h("div.muted", {}, "No merges yet."));

    // buttons
    btn.phase.disabled = playing || !canAdvance();
    btn.merge.disabled = playing || !canAdvance();
    btn.all.disabled = playing || exhausted || trainer.merges.length >= limit();
    btn.play.textContent = playing ? "⏸ Pause" : "▶ Play";
    btn.play.disabled = !playing && !canAdvance();
  }

  reset();
}
