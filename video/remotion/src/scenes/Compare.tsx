import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import data from "../data.json";
import { Backdrop, C, Chip, MONO, Title, useFade, usePop } from "../theme";

export const COMPARE_FRAMES = 360;

type Row = { label: string; value: number; shown: string; detail?: string };

const Column: React.FC<{ title: string; rule: string; rows: Row[]; start: number; color: string }> = ({ title, rule, rows, start, color }) => {
  const frame = useCurrentFrame();
  const max = Math.max(...rows.map((r) => r.value));
  return (
    <div style={{ flex: 1, background: C.panel, border: `1px solid ${C.line}`, borderRadius: 24, padding: "36px 40px", opacity: useFade(start - 10, start + 10) }}>
      <div style={{ fontSize: 44, fontWeight: 800, color }}>{title}</div>
      <div style={{ fontFamily: MONO, fontSize: 28, color: C.muted, marginTop: 8, minHeight: 44 }}>{rule}</div>
      <div style={{ marginTop: 28 }}>
        {rows.map((r, i) => {
          const p = interpolate(frame, [start + 20 + i * 8, start + 50 + i * 8], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
          const win = i === 0 && frame > start + 110;
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 18, padding: "10px 0", opacity: Math.min(1, p * 2) }}>
              <div style={{ width: 300, fontFamily: MONO, fontSize: 30, color: win ? C.win : C.ink, fontWeight: win ? 800 : 400 }}>{r.label}</div>
              <div style={{ flex: 1, height: 30, background: "rgba(255,255,255,0.05)", borderRadius: 8 }}>
                <div style={{ width: `${(r.value / max) * 100 * p}%`, height: "100%", borderRadius: 8, background: win ? C.win : color, opacity: win ? 1 : 0.55 }} />
              </div>
              <div style={{ width: 150, textAlign: "right", fontFamily: MONO, fontSize: 28, color: win ? C.win : C.muted }}>{r.shown}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const Compare: React.FC = () => {
  const frame = useCurrentFrame();
  const bpe: Row[] = data.compare.bpe.map((r) => ({ label: r.pair.join(" + "), value: r.count, shown: `${r.count}×` }));
  const wp: Row[] = data.compare.wordpiece.map((r) => ({
    label: r.pair.join(" + "),
    value: r.score,
    shown: r.score.toFixed(3),
  }));
  const out = interpolate(frame, [COMPARE_FRAMES - 12, COMPARE_FRAMES], [1, 0], { extrapolateLeft: "clamp" });
  const note = useFade(215, 250);
  const w = data.compare.wordpiece[0];
  return (
    <Backdrop>
      <AbsoluteFill style={{ padding: "70px 100px", opacity: out }}>
        <div style={{ opacity: useFade(0, 18), transform: `translateY(${(1 - usePop(0)) * 20}px)` }}>
          <div style={{ color: C.accent, fontSize: 26, fontWeight: 700, letterSpacing: 5 }}>BPE vs WORDPIECE</div>
          <Title size={70} style={{ marginTop: 10 }}>Same merge loop. Different score.</Title>
        </div>
        <div style={{ display: "flex", gap: 40, marginTop: 44 }}>
          <Column title="BPE" rule="most frequent pair wins" rows={bpe} start={30} color={C.accent} />
          <Column title="WordPiece" rule="count(ab) ÷ ( count(a) · count(b) )" rows={wp} start={60} color={C.warn} />
        </div>
        <div style={{ marginTop: 34, opacity: note, fontSize: 36, color: C.muted, maxWidth: 1650, lineHeight: 1.35 }}>
          Same toy corpus, different first merge. WordPiece rewards pairs whose parts rarely appear apart —{" "}
          <span style={{ fontFamily: MONO, color: C.warn }}>
            {w.pair.join(" + ")} = {w.count} ÷ ({w.ca} × {w.cb}) = {w.score.toFixed(3)}
          </span>
          . Encoding differs too: BPE replays merges; WordPiece does greedy longest-match with <Chip text="##" size={28} /> pieces.
        </div>
      </AbsoluteFill>
    </Backdrop>
  );
};
