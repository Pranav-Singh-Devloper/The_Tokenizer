import React from "react";
import { AbsoluteFill, OffthreadVideo, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { Backdrop, C, SANS } from "../theme";

export type Timeline = { duration: number; captions: { t: number; text: string }[] };

/** A Manim-rendered clip with captions synced to markers the scene wrote while rendering. */
export const ManimClip: React.FC<{ file: string; timeline: Timeline; badge: string; frames: number }> = ({
  file,
  timeline,
  badge,
  frames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const idx = timeline.captions.reduce((acc, c, i) => (c.t <= t ? i : acc), -1);
  const cap = idx >= 0 ? timeline.captions[idx] : null;
  const since = cap ? (t - cap.t) * fps : 0;
  const capIn = interpolate(since, [0, 10], [0, 1], { extrapolateRight: "clamp" });
  const edge = interpolate(frame, [0, 10, frames - 10, frames], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <Backdrop>
      <AbsoluteFill style={{ opacity: edge }}>
        <OffthreadVideo src={staticFile(`manim/${file}.mp4`)} style={{ width: "100%", height: "100%" }} />
        <div
          style={{
            position: "absolute",
            top: 36,
            right: 48,
            padding: "8px 18px",
            borderRadius: 999,
            background: "rgba(139,147,255,0.16)",
            color: C.accent,
            fontFamily: SANS,
            fontWeight: 700,
            fontSize: 24,
            letterSpacing: 1,
          }}
        >
          {badge}
        </div>
        {cap && (
          <div
            style={{
              position: "absolute",
              left: 0,
              right: 0,
              bottom: 44,
              display: "flex",
              justifyContent: "center",
              opacity: capIn,
              transform: `translateY(${(1 - capIn) * 14}px)`,
            }}
          >
            <div
              style={{
                maxWidth: 1500,
                textAlign: "center",
                fontFamily: SANS,
                fontSize: 38,
                lineHeight: 1.3,
                padding: "14px 30px",
                borderRadius: 16,
                background: "rgba(14,16,21,0.82)",
                border: `1px solid ${C.line}`,
                color: C.ink,
              }}
            >
              {cap.text}
            </div>
          </div>
        )}
      </AbsoluteFill>
    </Backdrop>
  );
};
