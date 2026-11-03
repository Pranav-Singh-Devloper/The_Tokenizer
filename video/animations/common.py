"""Shared look-and-feel for the Manim scenes (matches the web visualizer)."""
from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

from manim import *  # noqa: F401,F403

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

BG, PANEL, INK, MUTED = "#0e1015", "#161a22", "#e8e9ec", "#9298a5"
ACCENT, WIN, WARN, BAD = "#8b93ff", "#4ade80", "#fbbf24", "#f87171"
SANS, MONO = "Helvetica Neue", "Menlo"
OUT = Path(__file__).resolve().parent / "timelines"


def hue(key: str) -> int:
    """Same stable colour-per-token function as web/js/util.js."""
    x = 0
    for ch in str(key):
        x = (x * 31 + ord(ch)) & 0xFFFFFFFF
    return (x * 47) % 360


def hsl(h: float, s: float, l: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h / 360, l, s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def vis(s: str) -> str:
    return s.replace(" ", "·").replace("\n", "↵")


def chip(tok: str, fs: int = 24, color: str | None = None) -> VGroup:
    """A coloured token chip: rounded rectangle + monospace label."""
    h = hue(tok)
    label = Text(vis(tok), font=MONO, font_size=fs, color=INK)
    box = RoundedRectangle(
        corner_radius=0.09, width=max(label.width + 0.4, 0.5), height=0.56,
        fill_color=color or hsl(h, 0.55, 0.27), fill_opacity=1, stroke_color=hsl(h, 0.55, 0.5), stroke_width=2,
    )
    label.move_to(box)
    return VGroup(box, label)


def label(text: str, fs: int = 26, color: str = INK, weight=NORMAL, font: str = SANS) -> Text:
    return Text(text, font=font, font_size=fs, color=color, weight=weight)


class Timeline:
    """Records `(time, caption)` markers while a scene renders; Remotion reads the
    JSON to show captions synced to the animation, whatever the render speed."""

    def __init__(self, scene: Scene, name: str):
        self.scene, self.name, self.marks = scene, name, []

    def mark(self, caption: str) -> None:
        self.marks.append({"t": round(self.scene.renderer.time, 3), "text": caption})

    def save(self) -> None:
        OUT.mkdir(exist_ok=True)
        (OUT / f"{self.name}.json").write_text(
            json.dumps({"duration": round(self.scene.renderer.time, 3), "captions": self.marks}, ensure_ascii=False, indent=1)
        )
