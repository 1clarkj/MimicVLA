"""Tiny repeatable tracking demonstration without an attached camera."""

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from .track import track_video


def run_smoke(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "synthetic_calibration.json"
    calibration_path.write_text(json.dumps({
        "schema": "mimicvla.camera_calibration.v1",
        "source": "synthetic, not an iPhone calibration",
        "image_size_wh": [640, 480],
        "camera_matrix": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
        "distortion_coefficients": [0, 0, 0, 0, 0],
    }, indent=2) + "\n")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    usb = cv2.aruco.generateImageMarker(dictionary, 0, 60)
    port = cv2.aruco.generateImageMarker(dictionary, 1, 150)
    video_path = output_dir / "synthetic_markers.mp4"
    with imageio.get_writer(video_path, fps=10, codec="libx264", quality=9) as writer:
        for index in range(20):
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)
            frame[150:300, 100:250] = cv2.cvtColor(port, cv2.COLOR_GRAY2RGB)
            if index not in (7, 8):
                left = 370 + 2 * index
                frame[180:240, left:left + 60] = cv2.cvtColor(usb, cv2.COLOR_GRAY2RGB)
            writer.append_data(frame)
    return track_video(str(video_path), calibration_path, output_dir / "tracked",
                       usb_marker_m=0.020)
