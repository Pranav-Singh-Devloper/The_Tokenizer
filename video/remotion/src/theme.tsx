import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

export const C = {
  bg: "#0e1015",
  panel: "#161a22",
  line: "#272c37",
  ink: "#e8e9ec",
  muted: "#9298a5",
  accent: "#8b93ff",
  win: "#4ade80",
  warn: "#fbbf24",
  bad: "#f87171",
};
export const SANS = '"Helvetica Neue", system-ui, -apple-system, "Segoe UI", sans-serif';
export const MONO = 'Menlo, "SF Mono", ui-monospace, monospace';

/** Same stable colour-per-token function as the web visualizer and the Manim scenes. */
export const hue = (key: string): number => {
  let x = 0;
  for (const ch of key) x = (Math.imul(x, 31) + (ch.codePointAt(0) ?? 0)) >>> 0;
  return (x * 47) % 360;
};

export const Backdrop: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div
    style={{
      position: "absolute",
      inset: 0,
      background: `radial-gradient(1200px 700px at 20% 0%, #1a1f33 0%, ${C.bg} 60%)`,
      fontFamily: SANS,
      color: C.ink,
    }}
  >
    {children}
  </div>
);

/** Token chip, visually identical to the website's. */
export const Chip: React.FC<{ text: string; id?: number; size?: number; style?: React.CSSProperties }> = ({
  text,
  id,
  size = 40,
  style,
}) => {
  const h = hue(text);
  const shown = text.replace(/ /g, "·").replace(/\n/g, "↵");
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "baseline",
        gap: 6,
        fontFamily: MONO,
        fontSize: size,
        lineHeight: 1.25,
        padding: `${size * 0.1}px ${size * 0.28}px`,
        borderRadius: size * 0.25,
        background: `hsl(${h} 55% 27%)`,
        border: `2px solid hsl(${h} 55% 50%)`,
        whiteSpace: "pre",
        ...style,
      }}
    >
      {shown}
      {id !== undefined && <sub style={{ fontSize: size * 0.42, color: C.muted }}>{id}</sub>}
    </span>
  );
};

/** Pop-in animation helper: 0 → 1 with a spring, starting at `delay` frames. */
export const usePop = (delay = 0, damping = 14) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return spring({ frame: frame - delay, fps, config: { damping, stiffness: 120 } });
};

export const useFade = (from: number, to: number) => {
  const frame = useCurrentFrame();
  return interpolate(frame, [from, to], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
};

export const Title: React.FC<{ children: React.ReactNode; size?: number; style?: React.CSSProperties }> = ({
  children,
  size = 84,
  style,
}) => (
  <div style={{ fontSize: size, fontWeight: 800, letterSpacing: -2, lineHeight: 1.05, ...style }}>{children}</div>
);
