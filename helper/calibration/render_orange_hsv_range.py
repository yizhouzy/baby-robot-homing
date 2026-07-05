"""Render the HSV hue range selected by robot_control.vision.isolate_orange.

Examples:
    uv run --project ariel python helper/render_orange_hsv_range.py
    uv run --project ariel python helper/render_orange_hsv_range.py --output results/vision_calibration/orange_range.png
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_control.vision import ORANGE_HSV_RANGES, isolate_orange


OUTPUT_DIR = Path("results/vision_calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the color range selected by isolate_orange()."
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--saturation", type=int, default=255)
    return parser.parse_args()


def hue_spectrum(width: int, row_height: int) -> np.ndarray:
    hue = np.linspace(0, 179, width, dtype=np.uint8)
    hsv_row = np.zeros((row_height, width, 3), dtype=np.uint8)
    hsv_row[:, :, 0] = hue[None, :]
    hsv_row[:, :, 1] = 255
    hsv_row[:, :, 2] = 255
    return cv2.cvtColor(hsv_row, cv2.COLOR_HSV2RGB)


def hue_value_grid(width: int, grid_height: int, saturation: int) -> np.ndarray:
    hue = np.linspace(0, 179, width, dtype=np.uint8)
    value = np.linspace(255, 0, grid_height, dtype=np.uint8)
    hsv = np.zeros((grid_height, width, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue[None, :]
    hsv[:, :, 1] = np.uint8(np.clip(saturation, 0, 255))
    hsv[:, :, 2] = value[:, None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def draw_hue_ticks(image: np.ndarray, y: int, width: int) -> None:
    hues = sorted({bound[0] for hsv_range in ORANGE_HSV_RANGES for bound in hsv_range})
    for hue in hues:
        x = int(round(hue / 179 * (width - 1)))
        cv2.line(image, (x, y - 8), (x, y + 8), (255, 255, 255), 1)
        cv2.putText(
            image,
            str(hue),
            (max(0, x - 14), y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_value_ticks(image: np.ndarray, x: int, y: int, grid_height: int) -> None:
    for value in [255, 128, 50, 0]:
        row = y + int(round((255 - value) / 255 * (grid_height - 1)))
        cv2.line(image, (x - 8, row), (x + 8, row), (255, 255, 255), 1)
        cv2.putText(
            image,
            str(value),
            (x + 12, row + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def render_range_image(width: int, height: int, saturation: int) -> np.ndarray:
    margin = 18
    label_h = 28
    row_h = 42
    grid_h = max(80, (height - 6 * label_h - 2 * margin - 3 * row_h) // 2)
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)

    spectrum = hue_spectrum(width, row_h)
    mask = isolate_orange(spectrum)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    selected = (spectrum.astype(np.float32) * 0.22).astype(np.uint8)
    selected[mask > 0] = spectrum[mask > 0]

    hv_grid = hue_value_grid(width, grid_h, saturation)
    hv_mask = isolate_orange(hv_grid)
    hv_selected = (hv_grid.astype(np.float32) * 0.22).astype(np.uint8)
    hv_selected[hv_mask > 0] = hv_grid[hv_mask > 0]

    rows = [
        ("HSV hue spectrum, S=255, V=255", spectrum),
        ("Pixels kept by isolate_orange()", selected),
        ("Binary mask returned by isolate_orange()", mask_rgb),
        (f"Hue vs value, S={np.clip(saturation, 0, 255)}", hv_grid),
        ("Hue vs value kept by isolate_orange()", hv_selected),
    ]

    y = margin
    for label, row in rows:
        cv2.putText(
            canvas,
            label,
            (12, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        y += label_h
        h = row.shape[0]
        canvas[y:y + h, :] = row
        if label.startswith("HSV hue"):
            draw_hue_ticks(canvas, y + h - 12, width)
        if label.startswith("Hue vs value"):
            draw_hue_ticks(canvas, y + h - 12, width)
            draw_value_ticks(canvas, width - 58, y, h)
        y += h

    ranges = ", ".join(
        f"H {lower[0]}-{upper[0]}, S {lower[1]}-{upper[1]}, V {lower[2]}-{upper[2]}"
        for lower, upper in ORANGE_HSV_RANGES
    )
    note = f"Detected HSV ranges: {ranges}"
    cv2.putText(
        canvas,
        note,
        (12, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (240, 240, 240),
        1,
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

    image = render_range_image(args.width, args.height, args.saturation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    console.log(f"Saved orange HSV range visualization -> {output_path}")


if __name__ == "__main__":
    main()
