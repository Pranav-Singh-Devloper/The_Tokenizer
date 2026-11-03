#!/usr/bin/env bash
# Rebuild the explainer video end to end:
#   1. export real numbers from the library      -> remotion/src/data.json
#   2. render the Manim scenes (1080p30)          -> remotion/public/manim/*.mp4 + caption timelines
#   3. compose everything with Remotion           -> remotion/out/the-tokenizer.mp4
#
# Needs: python3 + manim, ffmpeg, node >= 18, and `python scripts/fetch_data.py` run once.
# If Remotion cannot download its headless browser, point it at an installed Chrome:
#   CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

python3 scripts/make_data.py

( cd animations && manim -r 1920,1080 --fps 30 scenes.py WordLevelProblem BPETraining EncodeUnseen )
M=animations/media/videos/scenes/1080p30
mkdir -p remotion/public/manim remotion/src/timelines
cp "$M/WordLevelProblem.mp4" remotion/public/manim/problem.mp4
cp "$M/BPETraining.mp4"      remotion/public/manim/bpe.mp4
cp "$M/EncodeUnseen.mp4"     remotion/public/manim/encode.mp4
cp animations/timelines/*.json remotion/src/timelines/

cd remotion
[ -d node_modules ] || npm install --no-audit --no-fund
EXTRA=()
[ -n "${CHROME:-}" ] && EXTRA+=(--browser-executable="$CHROME")
npx remotion render src/index.ts TokenizerExplainer out/the-tokenizer.mp4 --codec=h264 --crf=18 "${EXTRA[@]}"
echo "-> video/remotion/out/the-tokenizer.mp4"
