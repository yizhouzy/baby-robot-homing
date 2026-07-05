"""Render a compact HSV range swatch selected by robot_control.vision.isolate_orange.

Examples:
    uv run --project ariel python helper/calibration/render_orange_hsv_range.py
    uv run --project ariel python helper/calibration/render_orange_hsv_range.py --output results/vision_calibration/orange_range.png
"""
# ruff: noqa: E402
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import sys

import cv2
import numpy as np
from rich.console import Console
from rich.traceback import install

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_control.vision import ORANGE_HSV_RANGES, isolate_orange


OUTPUT_DIR = Path("results/vision_calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a compact color range swatch selected by isolate_orange()."
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=145)
    return parser.parse_args()


def hue_spectrum(width: int, row_height: int) -> np.ndarray:
    hue = np.linspace(0, 179, width, dtype=np.uint8)
    hsv_row = np.zeros((row_height, width, 3), dtype=np.uint8)
    hsv_row[:, :, 0] = hue[None, :]
    hsv_row[:, :, 1] = 255
    hsv_row[:, :, 2] = 255
    return cv2.cvtColor(hsv_row, cv2.COLOR_HSV2RGB)


def selected_hue_strip(width: int, height: int) -> np.ndarray:
    spectrum = hue_spectrum(width, height)
    mask = isolate_orange(spectrum)
    dimmed = (spectrum.astype(np.float32) * 0.18 + 26).astype(np.uint8)
    dimmed[mask > 0] = spectrum[mask > 0]
    return dimmed


def draw_selected_bounds(image: np.ndarray, x0: int, y0: int, width: int, height: int) -> None:
    for lower, upper in ORANGE_HSV_RANGES:
        left = x0 + int(round(lower[0] / 179 * (width - 1)))
        right = x0 + int(round(min(179, upper[0]) / 179 * (width - 1)))
        cv2.rectangle(image, (left, y0), (right, y0 + height - 1), (255, 255, 255), 1)
        cv2.line(image, (left, y0 + height), (left, y0 + height + 7), (235, 235, 235), 1)
        cv2.line(image, (right, y0 + height), (right, y0 + height + 7), (235, 235, 235), 1)


def render_range_image(width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), (245, 247, 249), dtype=np.uint8)
    pad = 12
    strip_w = width - 2 * pad
    full_y = 24
    kept_y = 68
    strip_h = 22

    canvas[full_y:full_y + strip_h, pad:pad + strip_w] = hue_spectrum(strip_w, strip_h)
    canvas[kept_y:kept_y + strip_h, pad:pad + strip_w] = selected_hue_strip(strip_w, strip_h)
    draw_selected_bounds(canvas, pad, kept_y, strip_w, strip_h)

    ranges = ", ".join(
        f"H {lower[0]}-{upper[0]}, S {lower[1]}-{upper[1]}, V {lower[2]}-{upper[2]}"
        for lower, upper in ORANGE_HSV_RANGES
    )
    labels = [
        ("HSV hue", full_y - 6),
        ("Retained Hue Band", kept_y - 6),
    ]
    for label, y in labels:
        cv2.putText(
            canvas,
            label,
            (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (75, 84, 94),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        ranges,
        (pad, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (48, 55, 63),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    install()
    console = Console()
    args = parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = OUTPUT_DIR / f"orange_hsv_range_{timestamp}.png"
    else:
        output_path = Path(args.output)

    image = render_range_image(args.width, args.height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    console.log(f"Saved orange HSV range visualization -> {output_path}")


if __name__ == "__main__":
    main()
