import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import data from "../data.json";
import { Backdrop, C, MONO, Title, useFade, usePop } from "../theme";

export const RESULTS_FRAMES = 660;
const CHART_END = 330;

const SERIES: { key: string; name: string; color: string; dash?: string }[] = [
  { key: "WordPiece-gain", name: "WordPiece (gain score)", color: "#66a61e" },
  { key: "char-BPE", name: "BPE (characters)", color: "#1b9e77" },
  { key: "byte-BPE", name: "Byte-level BPE", color: "#9a95e0" },
  { key: "WordPiece", name: "WordPiece (textbook score)", color: "#e7298a" },
  { key: "word", name: "word-level (hides 11% <unk>)", color: "#d95f02", dash: "10 8" },
];

const Chart: React.FC = () => {
  const frame = useCurrentFrame();
  const W = 1150, H = 640, m = { l: 90, r: 30, t: 20, b: 80 };
  const xs = [300, 500, 1000, 2000, 4000, 8000, 16000];
  const x = (v: number) => m.l + ((Math.log10(v) - Math.log10(300)) / (Math.log10(16000) - Math.log10(300))) * (W - m.l - m.r);
  const y0 = 1, y1 = 4.5;
  const y = (v: number) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
  return (
    <svg width={W} height={H} style={{ fontFamily: MONO }}>
      {[1, 2, 3, 4].map((g) => (
        <g key={g}>
          <line x1={m.l} x2={W - m.r} y1={y(g)} y2={y(g)} stroke={C.line} />
          <text x={m.l - 16} y={y(g) + 8} fill={C.muted} fontSize={24} textAnchor="end">{g}</text>
        </g>
      ))}
      {xs.map((v) => (
        <text key={v} x={x(v)} y={H - m.b + 38} fill={C.muted} fontSize={24} textAnchor="middle">{v >= 1000 ? `${v / 1000}k` : v}</text>
      ))}
      <text x={(m.l + W - m.r) / 2} y={H - 10} fill={C.muted} fontSize={26} textAnchor="middle" fontFamily="Helvetica Neue, sans-serif">vocabulary size (log scale)</text>
      <text x={24} y={H / 2 - 40} fill={C.muted} fontSize={26} textAnchor="middle" transform={`rotate(-90 24 ${H / 2 - 40})`} fontFamily="Helvetica Neue, sans-serif">characters per token →</text>
      {SERIES.map((s, si) => {
        const pts = (data.e1 as Record<string, number[][]>)[s.key];
        const d = pts.map(([v, c], i) => `${i ? "L" : "M"}${x(v)},${y(c)}`).join(" ");
        const p = interpolate(frame, [30 + si * 14, 150 + si * 14], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
        return (
          <g key={s.key}>
            <path d={d} fill="none" stroke={s.color} strokeWidth={6} strokeLinecap="round" pathLength={1} strokeDasharray={s.dash ?? "1"} strokeDashoffset={s.dash ? 0 : 1 - p} opacity={s.dash ? p : 1} />
            {pts.map(([v, c], i) => (
              <circle key={i} cx={x(v)} cy={y(c)} r={9} fill={s.color} opacity={p > i / (pts.length - 1) ? 1 : 0} />
            ))}
          </g>
        );
      })}
    </svg>
  );
};

const Stat: React.FC<{ big: string; small: string; color: string; delay: number }> = ({ big, small, color, delay }) => {
  const p = usePop(delay);
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 24, padding: "30px 40px", opacity: Math.min(1, p * 2), transform: `translateY(${(1 - p) * 40}px)` }}>
      <div style={{ fontSize: 96, fontWeight: 800, color, letterSpacing: -3, lineHeight: 1 }}>{big}</div>
      <div style={{ fontSize: 34, color: C.muted, marginTop: 12, lineHeight: 1.3 }}>{small}</div>
    </div>
  );
};

export const Results: React.FC = () => {
  const frame = useCurrentFrame();
  const f = data.facts;
  const out = interpolate(frame, [RESULTS_FRAMES - 12, RESULTS_FRAMES], [1, 0], { extrapolateLeft: "clamp" });
  const chartOp = interpolate(frame, [CHART_END - 15, CHART_END + 5], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const parity = Math.round(interpolate(frame, [CHART_END + 60, CHART_END + 130], [0, f.gpt2_parity_tokens], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }));
  return (
    <Backdrop>
      <AbsoluteFill style={{ padding: "60px 100px", opacity: out }}>
        <div style={{ opacity: useFade(0, 15) }}>
          <div style={{ color: C.accent, fontSize: 26, fontWeight: 700, letterSpacing: 5 }}>RESULTS · HELD-OUT SHAKESPEARE</div>
        </div>

        <div style={{ opacity: chartOp, display: "flex", gap: 50, alignItems: "center", marginTop: 12 }}>
          <Chart />
          <div style={{ flex: 1 }}>
            <Title size={56}>Compression vs vocabulary size</Title>
            <div style={{ marginTop: 30 }}>
              {SERIES.map((s, i) => (
                <div key={s.key} style={{ display: "flex", alignItems: "center", gap: 16, fontSize: 28, margin: "12px 0", opacity: useFade(40 + i * 14, 60 + i * 14) }}>
                  <span style={{ width: 36, height: 8, borderRadius: 4, background: s.color, display: "inline-block" }} />
                  {s.name}
                </div>
              ))}
            </div>
            <div style={{ marginTop: 28, fontSize: 28, color: C.muted, opacity: useFade(150, 180), lineHeight: 1.35 }}>
              The textbook WordPiece score trails far behind; weighting it by frequency flips it to the best compressor.
            </div>
          </div>
        </div>

        {frame >= CHART_END - 10 && (
          <div style={{ position: "absolute", left: 100, right: 100, top: 130 }}>
            <Title size={64}>What we measured</Title>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32, marginTop: 36 }}>
              <Stat big={`${(f.oov_rate * 100).toFixed(1)}%`} small="of held-out tokens are unknown to a word-level vocabulary — even with every training word" color={C.bad} delay={CHART_END + 5} />
              <Stat big={parity.toLocaleString()} small="tokens — identical to OpenAI's GPT-2 tokenizer, token for token" color={C.win} delay={CHART_END + 40} />
              <Stat big="1 of 5" small="tokenizers is lossless: byte-level BPE — decode(encode(x)) == x for any text" color={C.accent} delay={CHART_END + 100} />
              <Stat big={`${f.code_bpt["byte-BPE (V=2000, Shakespeare-trained)"].toFixed(2)} vs ${f.code_bpt["GPT-2 byte-BPE (V=50257, WebText-trained)"].toFixed(2)}`} small="bytes/token on Python code: a Shakespeare vocabulary vs GPT-2’s — a tokenizer encodes its training data" color={C.warn} delay={CHART_END + 160} />
            </div>
          </div>
        )}
      </AbsoluteFill>
    </Backdrop>
  );
};
