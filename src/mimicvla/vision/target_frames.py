"""Measured task points and provisional orientations in marker coordinates.

T_A_B maps coordinates in B into A. Lengths are metres. The socket offset is
the port-marker centre in the socket frame. The USB offset is the tip centre
in the detected USB-marker frame. Rotations still need visual confirmation.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .calibrate import load_calibration
from .transforms import inverse


def provisional_mounts() -> tuple[np.ndarray, np.ndarray]:
    """Return T_socket_port_marker and T_usb_marker_usb_tip."""
    port = np.eye(4)
    # Marker X follows socket X; printed Y and outward Z oppose socket Y
    # and inward Z. This places the target at the left USB-A port.
    port[:3, :3] = np.diag([1.0, -1.0, -1.0])
    port[:3, 3] = [0.016, -0.045, 0.0]

    marker_tip = np.eye(4)
    # Marker +Z is tip -Y; tip +Z (insertion) follows marker +Y.
    # The latter also follows the measured 30 mm marker-to-tip displacement.
    marker_tip[:3, :3] = np.array([[1., 0., 0.],
                                  [0., 0., 1.],
                                  [0., -1., 0.]])
    # User's corrected measurement: marker centre -> leading-face centre,
    # expressed along the OpenCV-detected marker axes.
    marker_tip[:3, 3] = [0.0, 0.030, -0.003]
    return port, marker_tip


def marker_tip_with_x_offset(x_mm: float) -> np.ndarray:
    """Return T_marker0_tip with a batch-specific X offset in marker axes."""
    _, marker_tip = provisional_mounts()
    marker_tip[0, 3] = x_mm / 1000
    return marker_tip


def socket_from_tip(record: dict, port_mount: np.ndarray,
                    marker_tip: np.ndarray) -> np.ndarray | None:
    """Convert a tracked marker pair into the task's relative target pose."""
    if record.get("quality") != "ok" or record.get("T_port_usb_marker") is None:
        return None
    return port_mount @ np.asarray(record["T_port_usb_marker"], dtype=float) @ marker_tip


def inspect_frame(trajectory: Path, calibration_path: Path, frame_index: int,
                  output: Path, *, tip_x_mm: float = 0.0) -> dict:
    """Draw proposed socket and tip frames on one original video frame."""
    records = [json.loads(line) for line in trajectory.read_text().splitlines()]
    record = records[frame_index]
    if record["quality"] != "ok":
        raise ValueError(f"Frame {frame_index} has quality {record['quality']}")
    frame_file = record.get("rgb_file")
    frame = cv2.imread(str(trajectory.parent / frame_file)) if frame_file else None
    if frame is None:
        metadata_file = trajectory.parent / "metadata.json"
        if not metadata_file.exists():
            raise ValueError("Original frame and tracking source metadata are missing")
        source = json.loads(metadata_file.read_text())["source"]
        capture = cv2.VideoCapture(source)
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            frame = None
    if frame is None:
        raise ValueError("Original frame is missing")
    calibration = load_calibration(calibration_path)
    camera = np.asarray(calibration["camera_matrix"], dtype=float)
    distortion = np.asarray(calibration["distortion_coefficients"], dtype=float)
    port_mount, _ = provisional_mounts()
    marker_tip = marker_tip_with_x_offset(tip_x_mm)
    markers = record["markers"]
    port_record = markers.get("port", markers.get("1"))
    if port_record is None:
        raise ValueError("Port marker pose is missing")
    camera_port = np.asarray(port_record["T_camera_marker"])
    usb_record = markers.get("usb")
    camera_usb = (np.asarray(usb_record["T_camera_marker"]) if usb_record is not None else
                  camera_port @ np.asarray(record["T_port_usb_marker"]))
    targets = {
        "socket_target": camera_port @ inverse(port_mount),
        "usb_tip": camera_usb @ marker_tip,
    }
    for name, transform in targets.items():
        rvec, _ = cv2.Rodrigues(transform[:3, :3])
        tvec = transform[:3, 3]
        cv2.drawFrameAxes(frame, camera, distortion, rvec, tvec, 0.012, 3)
        pixels, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, camera, distortion)
        x, y = np.rint(pixels[0, 0]).astype(int)
        colour = (0, 255, 0) if name == "socket_target" else (255, 0, 255)
        cv2.circle(frame, (x, y), 8, colour, 2)
        label = "socket" if name == "socket_target" else "USB tip"
        label_at = (x - 140, y - 22) if name == "socket_target" else (x + 12, y + 30)
        cv2.putText(frame, label, label_at, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4)
        cv2.putText(frame, label, label_at, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colour, 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise ValueError(f"Could not write {output}")
    relative = socket_from_tip(record, port_mount, marker_tip)
    return {"output": str(output), "frame_index": frame_index,
            "T_socket_usb_tip": relative.tolist(),
            "note": (f"USB tip position uses [{tip_x_mm:+g}, +30, -3] mm in detected "
                     "marker axes; orientation is provisional.")}
