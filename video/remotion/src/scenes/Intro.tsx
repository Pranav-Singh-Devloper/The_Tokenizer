import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import data from "../data.json";
import { Backdrop, C, Chip, MONO, Title, useFade, usePop } from "../theme";

export const INTRO_FRAMES = 240;

export const Intro: React.FC = () => {
  const frame = useCurrentFrame();
  const { sentence, tokens, ids } = data.intro;
  // typewriter: characters revealed between frame 40 and 110
  const n = Math.round(interpolate(frame, [40, 110], [0, sentence.length], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }));
  const typed = sentence.slice(0, n);
  const out = interpolate(frame, [INTRO_FRAMES - 12, INTRO_FRAMES], [1, 0], { extrapolateLeft: "clamp" });
  const subtitle = useFade(185, 215);

  return (
    <Backdrop>
      <AbsoluteFill style={{ opacity: out, padding: "90px 110px" }}>
        <div style={{ opacity: useFade(0, 20), transform: `translateY(${(1 - usePop(0)) * 24}px)` }}>
          <div style={{ color: C.accent, fontSize: 30, fontWeight: 700, letterSpacing: 6, textTransform: "uppercase" }}>The Tokenizer</div>
          <Title style={{ marginTop: 14 }}>How do language models read text?</Title>
        </div>

        <div style={{ marginTop: 70, fontFamily: MONO, fontSize: 50, minHeight: 70, color: C.ink }}>
          {typed}
          <span style={{ opacity: frame % 30 < 15 ? 1 : 0, color: C.accent }}>▌</span>
        </div>

        <div style={{ marginTop: 50, display: "flex", flexWrap: "wrap", gap: 14, maxWidth: 1700 }}>
          {tokens.map((t, i) => {
            const p = usePop(118 + i * 2.2);
            return (
              <div key={i} style={{ opacity: Math.min(1, p * 1.5), transform: `translateY(${(1 - p) * 30}px) scale(${0.8 + 0.2 * p})`, textAlign: "center" }}>
                <Chip text={t} size={42} />
                <div style={{ fontFamily: MONO, fontSize: 26, color: C.win, marginTop: 6, opacity: useFade(150 + i * 1.5, 170 + i * 1.5) }}>{ids[i]}</div>
              </div>
            );
          })}
        </div>

        <div style={{ position: "absolute", left: 110, right: 110, bottom: 90, opacity: subtitle, fontSize: 40, color: C.muted }}>
          Text → <b style={{ color: C.ink }}>tokens</b> → <b style={{ color: C.win }}>integer ids</b>. The tokenizer decides what the model can even represent.
          <div style={{ fontSize: 30, marginTop: 14 }}>Word-level, BPE, byte-level BPE and WordPiece — built from scratch.</div>
        </div>
      </AbsoluteFill>
    </Backdrop>
  );
};
