import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Backdrop, C, MONO, Title, useFade, usePop } from "../theme";

export const OUTRO_FRAMES = 240;

export const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const out = interpolate(frame, [OUTRO_FRAMES - 20, OUTRO_FRAMES], [1, 0], { extrapolateLeft: "clamp" });
  const p = usePop(0);
  return (
    <Backdrop>
      <AbsoluteFill style={{ padding: "120px 110px", opacity: out }}>
        <div style={{ opacity: Math.min(1, p * 2), transform: `translateY(${(1 - p) * 30}px)` }}>
          <Title size={110}>The Tokenizer</Title>
          <div style={{ fontSize: 44, color: C.muted, marginTop: 18 }}>Word-level & sub-word tokenizers, from scratch — with a research report and a live visualizer.</div>
        </div>

        <div style={{ marginTop: 70, display: "grid", gap: 26, fontSize: 42, opacity: useFade(40, 70) }}>
          <div><span style={{ color: C.accent, fontWeight: 700 }}>Live demo&nbsp;&nbsp;</span><span style={{ fontFamily: MONO }}>the-tokenizer.vercel.app</span></div>
          <div><span style={{ color: C.accent, fontWeight: 700 }}>Source&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style={{ fontFamily: MONO }}>github.com/Pranav-Singh-Devloper/The_Tokenizer</span></div>
        </div>

        <div style={{ marginTop: 70, display: "flex", gap: 22, flexWrap: "wrap", opacity: useFade(90, 120) }}>
          {["48 Python tests", "9 JS parity tests", "GPT-2 parity ✓", "zero-dependency core"].map((t) => (
            <div key={t} style={{ padding: "12px 26px", borderRadius: 999, border: `1px solid ${C.line}`, background: C.panel, fontSize: 30, color: C.win }}>{t}</div>
          ))}
        </div>

        <div style={{ position: "absolute", left: 110, bottom: 70, fontSize: 30, color: C.muted, opacity: useFade(120, 150) }}>
          Pranav Singh · animations: Manim · composition: Remotion
        </div>
      </AbsoluteFill>
    </Backdrop>
  );
};
