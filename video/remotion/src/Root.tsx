import React from "react";
import { Composition } from "remotion";
import { FPS, TOTAL_FRAMES, Video } from "./Video";

export const RemotionRoot: React.FC = () => (
  <Composition id="TokenizerExplainer" component={Video} durationInFrames={TOTAL_FRAMES} fps={FPS} width={1920} height={1080} />
);
