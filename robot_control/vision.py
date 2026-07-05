"""Target vision and robot-camera display helpers."""
from __future__ import annotations

import time

import cv2
import numpy as np


ORANGE_HSV_RANGES = (
    # ((5, 70, 50), (20, 255, 255)),
    ((157, 210, 230), (180, 255, 255)),
)

def isolate_orange(frame: np.ndarray) -> np.ndarray:
    """Return a binary mask for pixels inside the configured orange HSV range."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = None
    for lower, upper in ORANGE_HSV_RANGES:
        range_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = range_mask if mask is None else cv2.bitwise_or(mask, range_mask)
    return mask


def analyze_sections(mask: np.ndarray) -> list[float]:
    """Convert an HSV mask to five strip densities, centroid bearing, and area."""
    h, w = mask.shape
    strips = [
        float(cv2.countNonZero(section)) / section.size
        for section in np.array_split(mask, 5, axis=1)
    ]

    total_pixels = cv2.countNonZero(mask)
    if total_pixels > 0:
        moments = cv2.moments(mask)
        centroid_x = (moments["m10"] / moments["m00"] / w) * 2.0 - 1.0
    else:
        centroid_x = 0.0

    area = float(total_pixels) / float(h * w)
    return strips + [float(centroid_x), area]


def find_robot_camera(model) -> str | None:
    """Find the MuJoCo camera attached to the robot body."""
    for i in range(model.ncam):
        name = model.camera(i).name
        if ("camera" in name or "core" in name) and "video" not in name:
            return name
    return None


def sample_target_vision(renderer, data, cam_name: str | None):
    """Render the robot camera and extract the orange-target vision vector."""
    empty_frame = np.zeros((24, 32, 3), dtype=np.uint8)
    empty_mask = np.zeros((24, 32), dtype=np.uint8)
    if cam_name is None:
        return empty_frame, empty_mask, [0.0] * 7

    try:
        renderer.update_scene(data, camera=cam_name)
        camera_frame = renderer.render()
        mask = isolate_orange(camera_frame)
        return camera_frame, mask, analyze_sections(mask)
    except Exception:
        return empty_frame, empty_mask, [0.0] * 7


def open_camera(width: int, height: int):
    """Create a Picamera2 RGB stream at the requested resolution."""
    from picamera2 import Picamera2

    camera = Picamera2()
    config = camera.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
    )
    camera.configure(config)
    camera.start()
    time.sleep(0.5)
    return camera


def render_vision_pip(
    frame: np.ndarray,
    camera_frame: np.ndarray | None,
    mask: np.ndarray | None,
) -> None:
    """Draw robot-camera RGB and mask thumbnails in the rendered demo frame."""
    if camera_frame is None or mask is None:
        return
    camera_thumb = cv2.resize(camera_frame, (160, 120), interpolation=cv2.INTER_AREA)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask_thumb = cv2.resize(mask_rgb, (160, 120), interpolation=cv2.INTER_NEAREST)
    pip_frame = np.concatenate([camera_thumb, mask_thumb], axis=1)
    _, w, _ = frame.shape
    x0 = w - 330
    y0 = frame.shape[0] - 130
    frame[y0:y0 + 120, x0:x0 + 320] = pip_frame
    cv2.rectangle(frame, (x0, y0), (x0 + 320, y0 + 120), (220, 220, 220), 1)
    cv2.putText(
        frame,
        "Robot POV / target mask",
        (x0, y0 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
