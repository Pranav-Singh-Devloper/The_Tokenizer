// Encoding lab: watch a *trained* tokenizer process new text.
//   BPE        - replay the learned merges in rank order on one pre-token
//   WordPiece  - greedy longest-match-first window that shrinks until a piece is in the vocabulary
import { Trainer } from "./core.js";
import { h, clear, chip, visible, fmt, sleep } from "./util.js";
import { shared, loadSample } from "./state.js";

const DEMO_V = 800;
const SOURCES = {
  lab: "Model from the Training lab",
  "bpe-byte": "Demo · byte-level BPE trained on Shakespeare",
  "bpe-char": "Demo · character BPE trained on Shakespeare",
  wordpiece: "Demo · WordPiece trained on Shakespeare",
};
let api = null;

export const showEncodeLab = () => api?.refresh();

export function initEncodeLab(el) {
  let model = null, sel = 0, step = 0, playing = false, trace = null;

  const source = h("select", {}, Object.entries(SOURCES).map(([k, v]) => h("option", { value: k }, v)));
  const input = h("input", { type: "text", value: "lowest newer widest", style: { width: "100%" } });
  const info = h("span.small.muted");
  const ui = { pre: h("div.chips.pretoks"), stepper: h("div"), result: h("div") };

  el.append(
    h("h2", {}, "Encoding lab"),
    h("p.lead", {}, "Training produced a vocabulary and an ordered list of merges. Encoding unseen text means replaying that list — here you can watch it happen on a single pre-token, merge by merge."),
    h("div.card", {}, h("div.row", {}, h("label.f", {}, "Model", source), info),
      h("label.f", { style: { marginTop: "10px" } }, "Text to encode", input)),
    h("div.stage", {}, h("div.num", {}, "1"), h("div.card", {}, h("h3", {}, "Pre-tokenize"),
      h("p.small.muted", {}, "The text is first cut into word-like chunks. Learned merges are only applied inside a chunk. Click one to inspect it."), ui.pre)),
    h("div.stage", {}, h("div.num", {}, "2"), h("div.card", {}, h("h3", {}, "Apply the model to the selected chunk"), ui.stepper)),
    h("div.stage", {}, h("div.num", {}, "3"), h("div.card", {}, h("h3", {}, "Tokens and ids"), ui.result)));

  // if the user has not trained anything in the Training lab yet, start from a demo model
  source.value = shared.trainer ? "lab" : "bpe-byte";
  input.value = shared.trainer ? "lowest newer widest" : "unbelievably lowest tokenization";

  source.addEventListener("change", () => { sel = 0; refresh(); });
  input.addEventListener("input", () => { sel = 0; rebuild(); });

  async function getModel() {
    if (source.value === "lab") return shared.trainer?.model() ?? null;
    if (!shared.demoModels.has(source.value)) {
      info.textContent = `Training demo model on Shakespeare (V=${DEMO_V})…`;
      const text = await loadSample();
      const tr = new Trainer(text, { mode: source.value });
      await tr.runAsync(DEMO_V);
      shared.demoModels.set(source.value, tr.model());
    }
    return shared.demoModels.get(source.value);
  }

  async function refresh() {
    playing = false;
    model = await getModel();
    if (source.value === "lab" && model) info.textContent = `${shared.trainer.merges.length} merges, vocabulary ${model.vocabSize} (${shared.trainer.mode}).`;
    else if (model) info.textContent = `Vocabulary ${model.vocabSize}.`;
    rebuild();
  }

  function rebuild() {
    step = 0;
    clear(ui.pre); clear(ui.stepper); clear(ui.result);
    if (!model) { ui.stepper.append(h("p.muted", {}, "Train a model in the Training lab first.")); return; }
    const pres = model.pretokens(input.value);
    if (!pres.length) { ui.stepper.append(h("p.muted", {}, "Type some text.")); return; }
    sel = Math.min(sel, pres.length - 1);
    pres.forEach((p, i) => ui.pre.append(h("button", { class: i === sel ? "on" : "", onclick: () => { sel = i; rebuild(); } }, visible(p))));
    trace = model.mode === "wordpiece" ? model.traceWord(pres[sel]) : model.trace(pres[sel]);
    renderStepper();

    const toks = model.tokenize(input.value), ids = model.encode(input.value);
    ui.result.append(
      h("div.chips", {}, toks.slice(0, 300).map((t, i) => chip(t, { mode: model.mode, id: ids[i], cls: model.isUnk(t) ? "unk" : "" }))),
      h("p.small", { style: { marginTop: "10px" } }, h("b", {}, toks.length), " tokens · ", h("b", {}, fmt(Array.from(input.value).length / Math.max(toks.length, 1))), " chars/token"),
      h("p.small.mono", {}, "ids: [" + ids.slice(0, 80).join(", ") + (ids.length > 80 ? ", …" : "") + "]"),
      h("p.small", {}, "decoded: ", h("code", {}, JSON.stringify(model.decode(ids))),
        model.decode(ids) === input.value ? h("span.pill.ok", { style: { marginLeft: "8px" } }, "round-trip exact") : h("span.pill.warn", { style: { marginLeft: "8px" } }, "lossy")));
  }

  // ------------------------------------------------------------- BPE stepper
  function renderBpe() {
    const last = trace.length - 1, st = trace[step], nextPair = step < last ? trace[step + 1].pair : null;
    const syms = st.symbols.map((s, i) => {
      let cls = "";
      if (nextPair && s === nextPair[0] && st.symbols[i + 1] === nextPair[1]) cls = "pair-a";
      else if (nextPair && i > 0 && st.symbols[i - 1] === nextPair[0] && s === nextPair[1]) cls = "pair-b";
      else if (step > 0 && trace[step].pair && s === trace[step].pair[0] + trace[step].pair[1]) cls = "fresh";
      return chip(s, { mode: model.mode, cls });
    });
    const minRank = Math.min(...st.candidates.map((c) => c.rank ?? Infinity));
    const cands = st.candidates.length
      ? st.candidates.map((c) => h("div.cand", { class: c.rank === minRank && c.rank !== null ? "best" : "" },
        chip(c.pair[0], { mode: model.mode }), "+", chip(c.pair[1], { mode: model.mode }),
        h("span.rk", {}, c.rank === null ? "— not in table" : `merge rank #${c.rank + 1}`),
        c.rank === minRank && c.rank !== null ? "← lowest rank: applied next" : ""))
      : [h("div.muted.small", {}, "Only one symbol: nothing to merge.")];
    ui.stepper.append(
      h("p.small", {}, `State ${step} of ${last}. `, step === last
        ? h("b.done-note", {}, "No adjacent pair appears in the merge table — encoding of this chunk is finished.")
        : "Look at every adjacent pair, find the one learned earliest (lowest rank), merge all its occurrences."),
      h("div.chips", { style: { margin: "10px 0" } }, syms),
      h("div", {}, h("div.small.muted", {}, "Adjacent pairs and their rank in the merge table:"), cands));
  }

  // ------------------------------------------------------- WordPiece stepper
  function renderWp() {
    const log = trace.log, last = log.length;           // step counts tries shown
    const shown = log.slice(0, step);
    const pieces = [];
    shown.forEach((t) => { if (t.hit) pieces.push(t.piece); });
    const cur = shown.at(-1);
    const pre = model.pretokens(input.value)[sel];
    const chars = Array.from(pre);
    const chars_el = chars.map((c, i) => h("span.chip", { class: cur && i >= cur.start && i < cur.end ? (cur.hit ? "hit" : "pair-a") : "dim", style: { "--h": 210 } }, c));
    const done = step >= last;
    ui.stepper.append(
      h("p.small", {}, `Try ${step} of ${last}. `, done
        ? (trace.failed ? h("b", { style: { color: "var(--bad)" } }, "No prefix of the remaining text is in the vocabulary → the whole word becomes [UNK].")
          : h("b.done-note", {}, "Whole word consumed."))
        : "Take the longest remaining prefix; if it is not in the vocabulary, drop the last character and retry. After a hit, continue from there with a ## prefix."),
      h("div.chips", { style: { margin: "10px 0" } }, chars_el),
      h("div", {}, shown.map((t) => h("div.try", { class: t.hit ? "ok" : "no" }, h("span.m", {}, t.hit ? "✓" : "✗"), `"${t.piece}"`,
        h("span.muted", {}, t.hit ? "in vocabulary → emit" : "not in vocabulary")))),
      h("p.small", { style: { marginTop: "8px" } }, "pieces so far: ", h("span.chips", { style: { display: "inline-flex" } }, (done && trace.failed ? ["[UNK]"] : pieces).map((p) => chip(p, { mode: "wordpiece", cls: p === "[UNK]" ? "unk" : "" })))));
  }

  function renderStepper() {
    clear(ui.stepper);
    const wp = model.mode === "wordpiece";
    const max = wp ? trace.log.length : trace.length - 1;
    const play = h("button.btn.primary", { onclick: autoplay }, playing ? "⏸ Pause" : "▶ Auto");
    ui.stepper.append(h("div.row.tight", { style: { marginBottom: "10px" } },
      h("button.btn", { onclick: () => { step = 0; renderStepper(); }, disabled: step === 0 }, "⟲ Restart"),
      h("button.btn", { onclick: () => { step = Math.max(0, step - 1); renderStepper(); }, disabled: step === 0 }, "‹ Back"),
      h("button.btn", { onclick: () => { step = Math.min(max, step + 1); renderStepper(); }, disabled: step >= max }, wp ? "Next try ›" : "Apply merge ›"),
      play));
    wp ? renderWp() : renderBpe();
  }

  async function autoplay() {
    const max = model.mode === "wordpiece" ? trace.log.length : trace.length - 1;
    if (playing) { playing = false; renderStepper(); return; }
    if (step >= max) step = 0;
    playing = true; renderStepper();
    while (playing && step < max) { await sleep(750); if (!playing) break; step++; renderStepper(); }
    playing = false; renderStepper();
  }

  api = { refresh };
  refresh();
}
