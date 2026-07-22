#!/usr/bin/env python3
"""Generate a self-typing, monochrome ASCII portrait SVG.

Usage:
    python make_ascii_svg.py input.jpg ascii-portrait.svg
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove

RAMP = " .`:-=+*cs#%@"
COLS = 120
CLAHE_CLIP = 2.35
GAMMA = 1.28
CROP_BOTTOM = 0.08
FG_LIGHT = "#59636e"
FG_DARK = "#c9d1d9"
CHAR_W = 7.74
FONT_SIZE = 12.9
LINE_H = 15
ROW_DELAY = 0.075


def prepare_image(path: Path) -> Image.Image:
    """Remove the background and improve local facial contrast."""
    source = Image.open(path).convert("RGBA")
    cutout = remove(source)
    alpha = np.asarray(cutout.getchannel("A"))

    white = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
    gray = np.asarray(Image.alpha_composite(white, cutout).convert("L"))
    gray = cv2.bilateralFilter(gray, 11, 50, 50)
    gray = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=(8, 8),
    ).apply(gray)
    gray[alpha < 20] = 255
    return Image.fromarray(gray)


def image_to_lines(
    image: Image.Image,
    cols: int = COLS,
    gamma: float = GAMMA,
) -> list[str]:
    """Map image luminance to a fixed-width ASCII grid."""
    width, height = image.size
    if CROP_BOTTOM:
        image = image.crop((0, 0, width, int(height * (1 - CROP_BOTTOM))))
        width, height = image.size

    rows = max(1, round(cols * (height / width) * 0.48))
    image = image.resize((cols, rows), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    ramp_size = len(RAMP)

    lines: list[str] = []
    for row in pixels:
        indexes = np.minimum(
            ramp_size - 1,
            ((1 - row / 255.0) ** gamma * ramp_size).astype(int),
        )
        lines.append("".join(RAMP[index] for index in indexes).rstrip())

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def build_svg(lines: list[str], output_path: Path, cols: int = COLS) -> Path:
    """Write an SVG whose rows reveal from top to bottom once."""
    padding = 14
    width = int(cols * CHAR_W + padding * 2)
    height = len(lines) * LINE_H + padding * 2

    definitions: list[str] = [
        "<defs>",
        f"<style>.ascii{{fill:{FG_LIGHT}}}"
        f"@media(prefers-color-scheme:dark){{.ascii{{fill:{FG_DARK}}}}}</style>",
    ]
    artwork: list[str] = []

    for index, line in enumerate(lines):
        y = padding + index * LINE_H
        begin = index * ROW_DELAY
        end = (index + 1) * ROW_DELAY
        line_width = max(len(line), 1) * CHAR_W

        definitions.append(
            f'<clipPath id="row-{index}">'
            f'<rect x="{padding}" y="{y}" height="{LINE_H}" width="0">'
            f'<animate attributeName="width" from="0" to="{line_width:.1f}" '
            f'begin="{begin:.3f}s" dur="{ROW_DELAY:.3f}s" fill="freeze"/>'
            "</rect></clipPath>"
        )

        artwork.append(
            f'<g clip-path="url(#row-{index})">'
            f'<text xml:space="preserve" x="{padding}" y="{y + 11.2:.1f}" '
            f'class="ascii" font-size="{FONT_SIZE}">{escape(line)}</text></g>'
        )
        artwork.append(
            f'<rect y="{y + 1}" width="6" height="12" class="ascii" opacity="0">'
            f'<animate attributeName="x" from="{padding}" to="{padding + line_width:.1f}" '
            f'begin="{begin:.3f}s" dur="{ROW_DELAY:.3f}s" fill="freeze"/>'
            f'<set attributeName="opacity" to="0.8" begin="{begin:.3f}s"/>'
            f'<set attributeName="opacity" to="0" begin="{end:.3f}s"/>'
            "</rect>"
        )

    definitions.append("</defs>")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description" '
        'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">',
        "<title id=\"title\">ASCII portrait of Brian Rodriguez</title>",
        "<desc id=\"description\">A monochrome portrait that types itself one row at a time.</desc>",
        *definitions,
        *artwork,
        "</svg>",
    ]
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source portrait photo")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("ascii-portrait.svg"),
        help="output SVG path (default: ascii-portrait.svg)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = image_to_lines(prepare_image(args.source))
    if not lines:
        raise RuntimeError("No visible portrait remained after preprocessing")
    build_svg(lines, args.output)
    print("\n".join(lines))
    print(f"\nwrote {args.output} ({len(lines)} rows)")


if __name__ == "__main__":
    main()
