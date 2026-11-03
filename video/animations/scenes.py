"""Manim scenes for The Tokenizer explainer.

    manim -r 1920,1080 --fps 30 scenes.py WordLevelProblem BPETraining EncodeUnseen

Every number shown comes from running the real tokenlab library.
"""
from __future__ import annotations

from common import *  # noqa: F401,F403

from tokenlab import BPETokenizer, WordTokenizer
from tokenlab.bpe import _initial_symbols
from tokenlab.bytes_unicode import symbols_to_text
from tokenlab.merge_engine import MergeEngine

TOY = {"newest": 6, "low": 5, "widest": 3, "lower": 2}  # Sennrich et al. (2016)


# =========================================================================
class WordLevelProblem(Scene):
    """Why word-level vocabularies fail: the open-vocabulary problem."""

    def construct(self):
        self.camera.background_color = BG
        tl = Timeline(self, "problem")
        shake = (ROOT / "data" / "raw" / "shakespeare.txt").read_text(encoding="utf-8")
        words = WordTokenizer.train(shake[: int(len(shake) * 0.9)], max_size=20000)
        sentence = ["the", "king", "speaks", "of", "unbelievability"]
        ids = words.encode(" ".join(sentence))

        title = label("Word-level tokenizer", 40, INK, BOLD).to_edge(UP, buff=0.45)
        sub = label(f"vocabulary = every word in 1 MB of Shakespeare  ({words.vocab_size - 4:,} words)", 24, MUTED).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(title, shift=DOWN * 0.2), FadeIn(sub))
        tl.mark("A word-level tokenizer gives every known word its own id.")

        row = VGroup(*[chip(w, 30) for w in sentence]).arrange(RIGHT, buff=0.25).move_to(UP * 0.9)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in row], lag_ratio=0.2))
        self.wait(0.6)

        id_labels = VGroup()
        for c, i, w in zip(row, ids, sentence):
            known = w in words.vocab
            t = label(str(i) if known else "<unk>", 26, WIN if known else BAD, BOLD, MONO).next_to(c, DOWN, buff=0.55)
            arrow = Arrow(c.get_bottom(), t.get_top(), buff=0.08, stroke_width=3, color=WIN if known else BAD, max_tip_length_to_length_ratio=0.35)
            id_labels.add(VGroup(arrow, t))
        for k, item in enumerate(id_labels):
            self.play(FadeIn(item, shift=DOWN * 0.15), run_time=0.45)
            if k == len(sentence) - 2:
                tl.mark("…but a word it has never seen collapses into a single <unk> token.")
        bad = row[-1]
        self.play(Indicate(bad, color=BAD, scale_factor=1.15), bad[0].animate.set_stroke(BAD, 4), run_time=1.0)
        warn = label("information destroyed — the model cannot tell this word from any other unknown word", 24, BAD).move_to(DOWN * 1.15)
        self.play(FadeIn(warn, shift=UP * 0.1))
        self.wait(1.0)
        tl.mark("Even with EVERY training word in the vocabulary, 4.0% of held-out tokens are unknown.")
        stat = VGroup(label("4.0%", 64, BAD, BOLD), label("of held-out Shakespeare tokens are <unk>", 26, MUTED)).arrange(RIGHT, buff=0.3, aligned_edge=DOWN).move_to(DOWN * 2.2)
        self.play(FadeIn(stat, shift=UP * 0.2))
        self.wait(2.2)

        self.play(FadeOut(VGroup(row, id_labels, warn, stat, sub)))
        tl.mark("Sub-word tokenizers split rare words into reusable pieces instead.")
        q = label("What if a word could be built from pieces?", 36, INK).move_to(UP * 1.4)
        self.play(FadeIn(q, shift=UP * 0.15))
        bpe = BPETokenizer.train(shake[: int(len(shake) * 0.9)], 2000, byte_level=True)
        pieces = [symbols_to_text(p) for p in bpe.tokenize(" unbelievability")]  # bytes -> readable text
        word = chip("unbelievability", 40).move_to(UP * 0.0)
        self.play(FadeIn(word, shift=UP * 0.2))
        self.wait(0.6)
        chips = VGroup(*[chip(p, 36) for p in pieces]).arrange(RIGHT, buff=0.12).move_to(DOWN * 1.1)
        self.play(TransformFromCopy(word, chips[0]), run_time=0.01)
        self.play(ReplacementTransform(word, chips), run_time=1.6)
        note = label("byte-level BPE · 2,000-token vocabulary · nothing is ever unknown", 24, WIN).next_to(chips, DOWN, buff=0.6)
        self.play(FadeIn(note))
        self.wait(2.0)
        tl.save()


# =========================================================================
class BPETraining(Scene):
    """The BPE merge loop on the Sennrich toy corpus: count, choose, merge."""

    N_MERGES = 6
    X0, GAP = -6.0, 0.07
    ROW_Y = [1.55, 0.6, -0.35, -1.3]

    def precompute(self):
        words = {_initial_symbols(w, False): c for w, c in TOY.items()}
        eng = MergeEngine(words)
        order = sorted(range(len(eng.seqs)), key=lambda i: -eng.freqs[i])
        self.order = order
        self.freqs = [eng.freqs[i] for i in order]
        self.start = [list(eng.seqs[i]) for i in order]
        self.steps = []
        for _ in range(self.N_MERGES):
            top = sorted(eng.pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            before = [list(eng.seqs[i]) for i in order]
            pair = eng.best_by_frequency()
            eng.merge(pair, pair[0] + pair[1])
            after = [list(eng.seqs[i]) for i in order]
            self.steps.append(dict(top=top, pair=pair, new=pair[0] + pair[1], before=before, after=after))

    def row_chips(self, syms, y):
        chips = [chip(s, 22) for s in syms]
        x = self.X0
        for c in chips:
            c.move_to([x + c.width / 2, y, 0])
            x += c.width + self.GAP
        return chips

    def pair_table(self, top, winner=None):
        g = VGroup()
        mx = top[0][1]
        for k, ((a, b), n) in enumerate(top):
            y = 1.75 - k * 0.52
            name = label(f"{vis(a)} + {vis(b)}", 20, INK, NORMAL, MONO).move_to([2.0, y, 0], aligned_edge=LEFT)
            bar = Rectangle(width=max(0.06, 1.75 * n / mx), height=0.26, fill_color=WIN if (a, b) == winner else ACCENT,
                            fill_opacity=0.85 if (a, b) == winner else 0.45, stroke_width=0).move_to([4.55, y, 0], aligned_edge=LEFT)
            cnt = label(str(n), 20, WIN if (a, b) == winner else MUTED, BOLD, MONO).move_to([6.95, y, 0], aligned_edge=RIGHT)
            g.add(VGroup(name, bar, cnt))
        return g

    def construct(self):
        self.camera.background_color = BG
        tl = Timeline(self, "bpe")
        self.precompute()
        title = label("Byte Pair Encoding · training", 36, INK, BOLD).to_corner(UL, buff=0.45)
        self.play(FadeIn(title, shift=RIGHT * 0.2))
        tl.mark("Start with a tiny corpus. Every word is split into characters.")

        # left: corpus
        head_l = label("Corpus  (frequency × symbols)", 22, MUTED).move_to([-3.4, 2.55, 0])
        freq_lbls = [label(f"{f}×", 22, MUTED, NORMAL, MONO).move_to([-6.6, y, 0]) for f, y in zip(self.freqs, self.ROW_Y)]
        self.rows = [self.row_chips(s, y) for s, y in zip(self.start, self.ROW_Y)]
        self.play(FadeIn(head_l), *[FadeIn(f) for f in freq_lbls], *[LaggedStart(*[FadeIn(c) for c in r], lag_ratio=0.08) for r in self.rows], run_time=1.6)

        head_r = label("Adjacent pair counts", 22, MUTED).move_to([4.6, 2.55, 0])
        head_v = label("Vocabulary additions", 22, MUTED).move_to([4.15, -1.15, 0])
        counter = label("merge 0", 26, ACCENT, BOLD, MONO).next_to(title, RIGHT, buff=0.7)
        vlen = lambda: sum(len(r) for r in self.rows) and sum(len(r) * f for r, f in zip(self.rows, self.freqs))
        csym = label(f"corpus: {vlen()} symbols", 22, MUTED, NORMAL, MONO).move_to([-3.4, -2.55, 0])
        self.play(FadeIn(head_r), FadeIn(head_v), FadeIn(counter), FadeIn(csym))
        self.wait(0.5)

        vocab_chips, table = [], None
        for k, st in enumerate(self.steps):
            fast = k >= 2
            rt = 0.55 if fast else 1.0
            newc = label(f"merge {k + 1}", 26, ACCENT, BOLD, MONO).move_to(counter, aligned_edge=LEFT)
            self.play(Transform(counter, newc), run_time=0.25)

            # ---- 1. count
            if k == 0: tl.mark("Phase 1 · COUNT every adjacent pair, weighted by how often its word occurs.")
            new_table = self.pair_table(st["top"])
            if table is None:
                self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in new_table], lag_ratio=0.12), run_time=rt * 1.3)
            else:
                self.play(FadeOut(table, run_time=0.2), LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in new_table], lag_ratio=0.1), run_time=rt * 1.1)
            table = new_table
            if not fast: self.wait(0.8)

            # ---- 2. choose
            a, b = st["pair"]
            if k == 0: tl.mark("Phase 2 · CHOOSE the most frequent pair: (e, s) occurs 9 times.")
            win_table = self.pair_table(st["top"], winner=(a, b))
            boxes = VGroup()
            for r, syms in zip(self.rows, st["before"]):
                for i in range(len(syms) - 1):
                    if syms[i] == a and syms[i + 1] == b:
                        boxes.add(SurroundingRectangle(VGroup(r[i], r[i + 1]), color=WARN, buff=0.06, stroke_width=3, corner_radius=0.1))
            win_box = SurroundingRectangle(win_table[0], color=WIN, buff=0.1, stroke_width=3, corner_radius=0.08)
            self.play(Transform(table, win_table), Create(win_box), *[Create(b_) for b_ in boxes], run_time=rt)
            if not fast: self.wait(1.0)

            # ---- 3. merge
            if k == 0: tl.mark("Phase 3 · MERGE: replace every (e, s) by the single new symbol “es”.")
            anims, new_rows, fresh = [], [], []
            for r, (old_s, new_s), y in zip(self.rows, zip(st["before"], st["after"]), self.ROW_Y):
                target = self.row_chips(new_s, y)
                out, i, j = [], 0, 0
                while i < len(old_s):
                    if i + 1 < len(old_s) and old_s[i] == a and old_s[i + 1] == b:
                        anims.append(ReplacementTransform(VGroup(r[i], r[i + 1]), target[j]))
                        out.append(target[j]); fresh.append(target[j]); i += 2
                    else:
                        anims.append(r[i].animate.move_to(target[j].get_center()))
                        out.append(r[i]); i += 1
                    j += 1
                new_rows.append(out)
            self.rows = new_rows
            vc = chip(st["new"], 22)
            vocab_chips.append(vc)
            # simple flow layout: wrap before a chip would cross the right margin
            x, y = 1.9, -1.8
            for c in vocab_chips:
                if x + c.width > 6.95: x, y = 1.9, y - 0.65
                if c is vc: break
                x += c.width + 0.12
            vc.move_to([x + vc.width / 2, y, 0])
            new_csym = label(f"corpus: {vlen()} symbols", 22, MUTED, NORMAL, MONO).move_to(csym)
            self.play(FadeOut(boxes), FadeOut(win_box), *anims, FadeIn(vc, scale=1.4), Transform(csym, new_csym), run_time=rt * 1.4)
            self.play(*[f[0].animate.set_stroke(WIN, 4) for f in fresh], run_time=0.3)
            self.play(*[f[0].animate.set_stroke(hsl(hue(st["new"]), 0.55, 0.5), 2) for f in fresh], run_time=0.3)
            if k == 1: tl.mark("Repeat. Each merge adds one symbol to the vocabulary and shortens the corpus.")
            if k == self.N_MERGES - 1: tl.mark("After 6 merges the whole word “newest” is a single token.")
            self.wait(0.3 if fast else 0.7)

        self.play(FadeOut(table))
        done = label("Frequent words become single tokens — rare words stay as reusable pieces.", 26, WIN).move_to([4.0, 0.4, 0])
        done.set_width(5.6)
        self.play(FadeIn(done, shift=UP * 0.1))
        self.wait(2.5)
        tl.save()


# =========================================================================
class EncodeUnseen(Scene):
    """Encoding replays the learned merges, lowest rank first."""

    def construct(self):
        self.camera.background_color = BG
        tl = Timeline(self, "encode")
        toy = " ".join(w for w, c in TOY.items() for _ in range(c))
        bpe = BPETokenizer.train(toy, 60, byte_level=False)
        states = bpe.trace_word("lowest")
        merges = bpe.merges

        title = label("Encoding an unseen word", 36, INK, BOLD).to_corner(UL, buff=0.45)
        self.play(FadeIn(title, shift=RIGHT * 0.2))
        tl.mark("The tokenizer never saw “lowest” in training. Can it still encode it?")

        # merge table (the model)
        head = label("Learned merge table", 22, MUTED).move_to([4.6, 2.75, 0])
        rows = VGroup()
        for r, (a, b) in enumerate(merges[:12]):
            t = label(f"#{r + 1:<2} {vis(a)} + {vis(b)}", 19, MUTED, NORMAL, MONO).move_to([2.1, 2.2 - r * 0.42, 0], aligned_edge=LEFT)
            rows.add(t)
        self.play(FadeIn(head), LaggedStart(*[FadeIn(t, shift=LEFT * 0.1) for t in rows], lag_ratio=0.05), run_time=1.5)

        def layout(syms):
            cs = [chip(s, 30) for s in syms]
            total = sum(c.width for c in cs) + 0.1 * (len(cs) - 1)
            x = -3.6 - total / 2
            for c in cs:
                c.move_to([x + c.width / 2, 0.6, 0]); x += c.width + 0.1
            return cs

        word = label("lowest", 54, INK, BOLD, MONO).move_to([-3.6, 2.0, 0])
        self.play(FadeIn(word, shift=DOWN * 0.2))
        chips = layout(states[0]["symbols"])
        self.play(ReplacementTransform(word.copy(), VGroup(*chips)), run_time=1.0)
        tl.mark("Start from single characters, then repeatedly apply the EARLIEST-learned merge that fits.")
        self.wait(0.8)

        for n, (prev, cur) in enumerate(zip(states, states[1:])):
            syms = prev["symbols"]
            ranks = [bpe.ranks.get((syms[i], syms[i + 1])) for i in range(len(syms) - 1)]
            rl = VGroup()
            for i, rk in enumerate(ranks):
                mid = (chips[i].get_right() + chips[i + 1].get_left()) / 2
                rl.add(label("–" if rk is None else f"#{rk + 1}", 22, MUTED if rk is None else ACCENT, BOLD, MONO).move_to(mid + DOWN * 0.8))
            self.play(FadeIn(rl, shift=UP * 0.1), run_time=0.5)
            if n == 0: tl.mark("Each neighbouring pair is looked up in the merge table; “–” means no rule.")
            best = cur["rank"]
            bi = ranks.index(best)
            row = rows[best]
            hl = SurroundingRectangle(VGroup(chips[bi], chips[bi + 1]), color=WARN, buff=0.07, stroke_width=3, corner_radius=0.1)
            rowbox = SurroundingRectangle(row, color=WIN, buff=0.06, stroke_width=2, corner_radius=0.06)
            self.play(Create(hl), Create(rowbox), rl[bi].animate.set_color(WIN), row.animate.set_color(WIN), run_time=0.6)
            if n == 0: tl.mark("The lowest rank wins: rule #1 turns e + s into “es”.")
            self.wait(0.5)
            target = layout(cur["symbols"])
            anims, newchips, i, j = [], [], 0, 0
            while i < len(syms):
                if i + 1 < len(syms) and [syms[i], syms[i + 1]] == cur["pair"]:
                    anims.append(ReplacementTransform(VGroup(chips[i], chips[i + 1]), target[j])); newchips.append(target[j]); i += 2
                else:
                    anims.append(chips[i].animate.move_to(target[j].get_center())); newchips.append(chips[i]); i += 1
                j += 1
            self.play(FadeOut(rl), FadeOut(hl), FadeOut(rowbox), *anims, run_time=0.9)
            chips = newchips
            self.wait(0.3)
        final = states[-1]["symbols"]
        ids = bpe.encode("lowest")
        tl.mark("No rule applies any more: “lowest” = low + est</w>. Both pieces are in the vocabulary, so we get ids.")
        idl = VGroup(*[label(str(i), 28, WIN, BOLD, MONO).next_to(c, DOWN, buff=0.5) for c, i in zip(chips, ids)])
        arrows = VGroup(*[Arrow(c.get_bottom(), l.get_top(), buff=0.06, stroke_width=3, color=WIN, max_tip_length_to_length_ratio=0.4) for c, l in zip(chips, idl)])
        self.play(FadeIn(arrows), FadeIn(idl, shift=DOWN * 0.1))
        out = label(f"“lowest” → {ids}", 30, INK, BOLD, MONO).move_to([-3.6, -1.9, 0])
        self.play(FadeIn(out, shift=UP * 0.1))
        self.wait(2.8)
        tl.save()
