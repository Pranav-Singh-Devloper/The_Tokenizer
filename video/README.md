# Explainer video

A ~2 minute explainer of the project, built with **Manim** (algorithm animations) and **Remotion**
(composition, titles, charts, synced captions). Every number on screen comes from running the
library or reading `results/summary.json`: nothing is typed in by hand.

| Segment | Tool | Content |
|---|---|---|
| Intro | Remotion | text → tokens → ids, using a real byte-level BPE trained on Shakespeare |
| 1 · The problem | Manim | word-level lookup, `<unk>`, the 4.0% held-out OOV rate, then sub-word pieces |
| 2 · BPE training | Manim | the Sennrich toy corpus: count → choose → merge, six merges |
| 3 · Encoding | Manim | replaying learned merges in rank order on the unseen word “lowest” |
| BPE vs WordPiece | Remotion | the two scoring rules on the same corpus |
| Results | Remotion | animated vocabulary-sweep chart + headline numbers |
| Outro | Remotion | links |

**How the two tools talk.** Each Manim scene records `(time, caption)` markers while it renders
(`animations/timelines/*.json`). Remotion reads those files, so the captions stay synced with the
animation and the clip lengths set the composition length automatically.

```bash
./video/build.sh                 # everything, end to end
cd video/remotion && npm run studio   # live preview / edit the Remotion scenes
cd video/animations && manim -pql scenes.py BPETraining   # iterate on one Manim scene
```

The rendered video is not committed (see `.gitignore`). It is hosted on Hugging Face: <https://huggingface.co/datasets/pranav-singh-developer/the-tokenizer-video> (direct file: <https://huggingface.co/datasets/pranav-singh-developer/the-tokenizer-video/resolve/main/the-tokenizer.mp4>).
