// JS port must reproduce the Python implementation exactly.
// Regenerate fixtures with: python scripts/export_fixtures.py
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Trainer, WordModel, BPEModel } from "../js/core.js";

const fx = JSON.parse(readFileSync(new URL("./fixtures.json", import.meta.url), "utf8"));
const asMap = (o) => new Map(Object.entries(o));

for (const c of fx.cases) {
  test(`${c.mode} V=${c.vocab_size}: training + tokenization match Python`, () => {
    if (c.mode === "word") {
      const m = WordModel.train(fx.corpus, c.vocab_size);
      assert.deepEqual(Object.fromEntries(m.vocab), c.vocab);
      fx.probes.forEach((p, i) => { assert.deepEqual(m.tokenize(p), c.tokens[i]); assert.deepEqual(m.encode(p), c.ids[i]); });
      return;
    }
    const tr = new Trainer(fx.corpus, { mode: c.mode }).run(c.vocab_size);
    if (c.merges) assert.deepEqual(tr.merges, c.merges, "merge list differs");
    // compare token->id as objects: JS orders integer-like keys ("0","1",..) first, ids are what matter
    assert.deepEqual(Object.fromEntries(tr.vocab), c.vocab, "vocab/ids differ");
    const m = tr.model();
    fx.probes.forEach((p, i) => {
      assert.deepEqual(m.tokenize(p), c.tokens[i], `tokens for ${JSON.stringify(p)}`);
      assert.deepEqual(m.encode(p), c.ids[i], `ids for ${JSON.stringify(p)}`);
    });
    if (c.traces) {
      for (const [w, py] of Object.entries(c.traces)) {
        const js = m.trace(w);
        assert.deepEqual(js.map((s) => s.symbols), py.map((s) => s.symbols), `trace of ${JSON.stringify(w)}`);
        assert.deepEqual(js.map((s) => s.rank), py.map((s) => s.rank));
      }
    }
  });
}

test("byte-level round trip is lossless for arbitrary text", () => {
  const c = fx.cases.find((x) => x.mode === "bpe-byte");
  const m = new BPEModel(asMap(c.vocab), c.merges, true);
  for (const s of ["", "héllo wörld 🐱", "  \n\t x", "日本語", "a".repeat(50)]) {
    assert.equal(m.decode(m.encode(s)), s);
  }
});

test("Sennrich toy example: first merges are (e,s) then (es,t</w>)", () => {
  const tr = new Trainer("low ".repeat(5) + "lower ".repeat(2) + "newest ".repeat(6) + "widest ".repeat(3), { mode: "bpe-char" });
  tr.step(); tr.step();
  assert.deepEqual(tr.merges, [["e", "s"], ["es", "t</w>"]]);
  assert.equal(tr.history[0].best.count, 9);
});
