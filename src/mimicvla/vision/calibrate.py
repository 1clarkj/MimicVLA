"""Calibrate the iPhone using the printed 5x7 ChArUco board."""

import json
from pathlib import Path

import cv2
import numpy as np


DICTIONARY = "DICT_4X4_50"
BOARD_SQUARES = (5, 7)
BOARD_IDS = np.arange(10, 27, dtype=np.int32)


def make_board(square_m: float = 0.030, marker_m: float = 0.022) -> cv2.aruco.CharucoBoard:
    if not 0 < marker_m < square_m:
        raise ValueError("Marker size must be positive and smaller than square size")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.CharucoBoard(BOARD_SQUARES, square_m, marker_m, dictionary, BOARD_IDS)


def image_samples(source: Path, stride: int):
    """Yield (name, BGR frame) from a directory of photos or a video."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for path in paths[::stride]:
            frame = cv2.imread(str(path))
            if frame is not None:
                yield path.name, frame
    else:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Cannot open calibration input: {source}")
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % stride == 0:
                    yield f"frame_{index:06d}", frame
                index += 1
        finally:
            capture.release()


def calibrate(source: Path, output: Path, *, square_m: float = 0.030,
              marker_m: float = 0.022, stride: int = 15, min_corners: int = 8) -> dict:
    board = make_board(square_m, marker_m)
    detector = cv2.aruco.CharucoDetector(board)
    object_points, image_points, accepted = [], [], []
    centers = []
    image_size = None
    for name, frame in image_samples(source, stride):
        size = (frame.shape[1], frame.shape[0])
        if image_size is not None and size != image_size:
            raise ValueError("Calibration frames have different resolutions")
        image_size = size
        corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None or len(ids) < min_corners:
            continue
        object_points.append(board.getChessboardCorners()[ids.ravel()].astype(np.float32))
        image_points.append(corners.reshape(-1, 2).astype(np.float32))
        accepted.append({"frame": name, "corners": int(len(ids))})
        centers.append(corners.reshape(-1, 2).mean(axis=0))
    if len(accepted) < 6:
        raise ValueError(f"Only {len(accepted)} useful views; record at least 6 varied board poses")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    center_span = np.ptp(np.asarray(centers), axis=0) / np.asarray(image_size)
    warnings = []
    if np.any(center_span < 0.30):
        warnings.append("Board centres cover less than 30% of image width or height; move the board toward image edges")
    half = len(accepted) // 2
    split_focal_difference = None
    if half >= 8 and len(accepted) - half >= 8:
        _, first_matrix, _, _, _ = cv2.calibrateCamera(
            object_points[:half], image_points[:half], image_size, None, None)
        _, second_matrix, _, _, _ = cv2.calibrateCamera(
            object_points[half:], image_points[half:], image_size, None, None)
        split_focal_difference = float(abs(first_matrix[0, 0] - second_matrix[0, 0]) / matrix[0, 0])
        if split_focal_difference > 0.05:
            warnings.append("First/second-half focal estimates differ by more than 5%; collect more varied board tilts and distances")
    result = {
        "schema": "mimicvla.camera_calibration.v1",
        "source": str(source),
        "image_size_wh": list(image_size),
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.ravel().tolist(),
        "rms_reprojection_px": float(rms),
        "board": {"dictionary": DICTIONARY, "squares_xy": list(BOARD_SQUARES),
                  "ids": BOARD_IDS.tolist(), "square_m": square_m, "marker_m": marker_m},
        "accepted_views": accepted,
        "board_center_span_fraction_xy": center_span.tolist(),
        "split_focal_difference_fraction": split_focal_difference,
        "warnings": warnings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_calibration(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema") != "mimicvla.camera_calibration.v1":
        raise ValueError("Unsupported camera calibration file")
    if np.asarray(data["camera_matrix"]).shape != (3, 3):
        raise ValueError("Camera matrix must be 3x3")
    if len(data["image_size_wh"]) != 2:
        raise ValueError("Calibration must include image width and height")
    return data
