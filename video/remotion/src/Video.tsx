import React from "react";
import { Series } from "remotion";
import problem from "./timelines/problem.json";
import bpe from "./timelines/bpe.json";
import encode from "./timelines/encode.json";
import { Compare, COMPARE_FRAMES } from "./scenes/Compare";
import { Intro, INTRO_FRAMES } from "./scenes/Intro";
import { ManimClip } from "./scenes/ManimClip";
import { Outro, OUTRO_FRAMES } from "./scenes/Outro";
import { Results, RESULTS_FRAMES } from "./scenes/Results";

export const FPS = 30;
const clipFrames = (d: number) => Math.ceil(d * FPS) + 6; // small tail so the last Manim frame is held

const P = clipFrames(problem.duration);
const B = clipFrames(bpe.duration);
const E = clipFrames(encode.duration);

export const TOTAL_FRAMES = INTRO_FRAMES + P + B + E + COMPARE_FRAMES + RESULTS_FRAMES + OUTRO_FRAMES;

export const Video: React.FC = () => (
  <Series>
    <Series.Sequence durationInFrames={INTRO_FRAMES}><Intro /></Series.Sequence>
    <Series.Sequence durationInFrames={P}>
      <ManimClip file="problem" timeline={problem} badge="1 · THE PROBLEM" frames={P} />
    </Series.Sequence>
    <Series.Sequence durationInFrames={B}>
      <ManimClip file="bpe" timeline={bpe} badge="2 · BPE TRAINING" frames={B} />
    </Series.Sequence>
    <Series.Sequence durationInFrames={E}>
      <ManimClip file="encode" timeline={encode} badge="3 · ENCODING" frames={E} />
    </Series.Sequence>
    <Series.Sequence durationInFrames={COMPARE_FRAMES}><Compare /></Series.Sequence>
    <Series.Sequence durationInFrames={RESULTS_FRAMES}><Results /></Series.Sequence>
    <Series.Sequence durationInFrames={OUTRO_FRAMES}><Outro /></Series.Sequence>
  </Series>
);
