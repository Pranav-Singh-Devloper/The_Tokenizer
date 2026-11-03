import { h } from "./util.js";

export function initLearn(el) {
  el.append(
    h("div.hero", {},
      h("h1", {}, "How do language models read text?"),
      h("p.lead", {}, "A neural network cannot read letters — it reads integers. The tokenizer is the translator: it chops text into pieces (tokens) and maps each piece to an id. "
        + "It is the least glamorous and most consequential component of an LLM, and it is built from a surprisingly small idea."),
      h("div.row", {},
        h("a", { href: "#train", class: "btn primary cta" }, "▶ Watch BPE train, step by step"),
        h("a", { href: "#playground", class: "btn cta" }, "Compare tokenizers on your own text"))),

    h("div.card", {},
      h("h2", {}, "The pipeline"),
      h("div.pipeline", {},
        h("div.box", {}, h("b", {}, "1 · Normalise"), "Optional: lower-case, strip accents (BERT) — or none at all (GPT)."),
        h("div.arrow", {}, "→"),
        h("div.box", {}, h("b", {}, "2 · Pre-tokenize"), "Cut into word-like chunks. Learned merges never cross a chunk boundary."),
        h("div.arrow", {}, "→"),
        h("div.box", {}, h("b", {}, "3 · Model"), "Word lookup, BPE merges or WordPiece longest-match turn chunks into tokens."),
        h("div.arrow", {}, "→"),
        h("div.box", {}, h("b", {}, "4 · Ids"), "Each token string is looked up in the vocabulary → integer ids fed to the network."))),

    h("div.grid.g3", {},
      h("div.card", {},
        h("h3", {}, "Word-level"),
        h("p", {}, "One token per word. Short sequences, readable — but any word not seen in training becomes ", h("code", {}, "<unk>"), ". "
          + "Names, typos and inflections (“tokenizing” when only “tokenize” was seen) are lost. This is the ", h("b", {}, "open-vocabulary problem"), ".")),
      h("div.card", {},
        h("h3", {}, "BPE · byte-level BPE"),
        h("p", {}, "Start from characters (or raw bytes), repeatedly merge the most frequent adjacent pair. Frequent words become single tokens, rare words fall apart into reusable pieces. "
          + "On bytes, nothing is ever unknown. Used by GPT-2/3/4, Llama, …")),
      h("div.card", {},
        h("h3", {}, "WordPiece"),
        h("p", {}, "Same merge loop, different score: merge the pair that maximises ", h("code", {}, "count(ab) / (count(a)·count(b))"), ". "
          + "Encoding is greedy longest-match with ", h("code", {}, "##"), " continuation pieces. Used by BERT, DistilBERT, Electra."))),

    h("div.card", {},
      h("h2", {}, "Learning objectives"),
      h("ol.obj", {},
        h("li", {}, "Explain the open-vocabulary problem and why word-level vocabularies cannot solve it."),
        h("li", {}, "Trace the BPE training loop by hand: count pairs → pick the max → merge → repeat."),
        h("li", {}, "Explain why encoding replays merges in learned order, and why that makes BPE deterministic."),
        h("li", {}, "Contrast BPE (frequency) with WordPiece (likelihood ratio) and predict where they disagree."),
        h("li", {}, "Reason about the compression ↔ vocabulary-size ↔ sequence-length trade-off."),
        h("li", {}, "Understand why byte-level BPE is lossless and never emits ", h("code", {}, "<unk>"), "."))),

    h("div.card", {},
      h("h2", {}, "How to use this site"),
      h("ul", {},
        h("li", {}, h("b", {}, "Training lab"), " — the main simulation. Each merge is split into three phases (count, choose, merge); play it automatically or step through it."),
        h("li", {}, h("b", {}, "Encoding lab"), " — take the trained model and watch it tokenize a new word, merge by merge (or window by window for WordPiece)."),
        h("li", {}, h("b", {}, "Playground"), " — train all four tokenizers on a corpus and compare how they segment your text."),
        h("li", {}, h("b", {}, "Experiments"), " — results from the research study on 1 MB of Shakespeare: vocabulary-size sweeps, OOV behaviour and domain shift."))),
  );
}
