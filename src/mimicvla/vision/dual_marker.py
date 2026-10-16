"""Track a USB with marker 0 and a rigidly mounted marker 10.

The public relative pose remains T_port_usb_marker, where usb_marker means
marker 0, so the existing tip transform and replay code remain applicable.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .calibrate import load_calibration
from .target_frames import provisional_mounts
from .track import estimate_marker, plot_trajectory
from .transforms import inverse, translation_rotvec


DEFAULT_MARKER10_M = .022


def _frames(video: Path):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not np.isfinite(timestamp) or timestamp <= 0:
                timestamp = index / fps
            yield index, timestamp, frame
            index += 1
    finally:
        capture.release()


def _detector():
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def estimate_mount(videos: list[Path], calibration_path: Path, output: Path,
                   *, marker10_m: float = DEFAULT_MARKER10_M) -> dict:
    """Fit T_marker0_marker10 from co-visible frames, with robust trimming."""
    calibration = load_calibration(calibration_path)
    if marker10_m <= 0:
        raise ValueError("Marker 10 side length must be positive")
    marker_sizes = {0: .020, 1: .050, 10: marker10_m}
    camera = np.asarray(calibration["camera_matrix"])
    distortion = np.asarray(calibration["distortion_coefficients"])
    detector = _detector()
    transforms = []
    for video in videos:
        previous = {}
        for _, _, frame in _frames(video):
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            visible = {} if ids is None else {int(mid): corner for mid, corner in zip(ids.ravel(), corners)}
            poses = {}
            for marker_id in (0, 10):
                if marker_id not in visible:
                    continue
                estimate = estimate_marker(visible[marker_id], marker_sizes[marker_id],
                                           camera, distortion, previous.get(marker_id))
                if estimate is not None and estimate[1] <= 5:
                    poses[marker_id] = previous[marker_id] = estimate[0]
            if len(poses) == 2:
                transforms.append(inverse(poses[0]) @ poses[10])
    if len(transforms) < 20:
        raise ValueError(f"Need at least 20 shared marker observations; got {len(transforms)}")
    transforms = np.asarray(transforms)
    positions = transforms[:, :3, 3]
    rotations = Rotation.from_matrix(transforms[:, :3, :3])
    centre = np.median(positions, axis=0)
    orientation = rotations.mean()
    translation_error = np.linalg.norm(positions - centre, axis=1)
    rotation_error = (orientation.inv() * rotations).magnitude()
    keep = (translation_error < .015) & (rotation_error < np.deg2rad(15))
    if keep.sum() < 20:
        raise ValueError("The two marker poses do not support one rigid mount")
    mount = np.eye(4)
    mount[:3, :3] = rotations[keep].mean().as_matrix()
    mount[:3, 3] = np.median(positions[keep], axis=0)
    residual_position = np.linalg.norm(positions[keep] - mount[:3, 3], axis=1)
    residual_angle = (Rotation.from_matrix(mount[:3, :3]).inv() * rotations[keep]).magnitude()
    report = {
        "convention": "T_marker0_marker10 maps marker 10 coordinates into marker 0 coordinates",
        "T_marker0_marker10": mount.tolist(),
        "marker_sizes_m": {str(k): v for k, v in marker_sizes.items()},
        "marker10_size_note": "Outer black border measured as 22 mm; surrounding ChArUco cell is 30 mm",
        "shared_frames": len(transforms), "inlier_frames": int(keep.sum()),
        "translation_residual_mm_p50": float(np.median(residual_position) * 1000),
        "rotation_residual_deg_p50": float(np.rad2deg(np.median(residual_angle))),
        "source_videos": [str(video) for video in videos],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def track_dual_video(video: Path, calibration_path: Path, output_dir: Path,
                     mount_path: Path, *, use_second_marker: bool,
                     marker10_m: float = DEFAULT_MARKER10_M,
                     marker_tip: np.ndarray | None = None) -> dict:
    """Track with marker 0 only or with marker 10 plus marker 0 fallback."""
    calibration = load_calibration(calibration_path)
    camera = np.asarray(calibration["camera_matrix"])
    distortion = np.asarray(calibration["distortion_coefficients"])
    mount_report = json.loads(mount_path.read_text())
    if abs(mount_report["marker_sizes_m"]["10"] - marker10_m) > 1e-9:
        raise ValueError("Marker 10 size does not match the mount calibration")
    marker_sizes = {0: .020, 1: .050, 10: marker10_m}
    mount = np.asarray(mount_report["T_marker0_marker10"])
    detector = _detector()
    previous = {}
    missing_streak = {0: 0, 1: 0, 10: 0}
    port_mount, default_tip = provisional_mounts()
    if marker_tip is None:
        marker_tip = default_tip
    records = []
    for index, timestamp, frame in _frames(video):
        if [frame.shape[1], frame.shape[0]] != calibration["image_size_wh"]:
            raise ValueError(f"Video image size differs from calibration: {video}")
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        visible = {} if ids is None else {int(mid): corner for mid, corner in zip(ids.ravel(), corners)}
        poses = {}
        markers = {}
        for marker_id in (1, 0, 10):
            if marker_id not in visible or (marker_id == 10 and not use_second_marker):
                missing_streak[marker_id] += 1
                if missing_streak[marker_id] > 4:
                    previous.pop(marker_id, None)
                continue
            expected = None
            if marker_id == 0 and 1 in poses:
                expected = poses[1] @ inverse(port_mount) @ inverse(marker_tip)
            elif marker_id == 10:
                if 0 in poses:
                    expected = poses[0] @ mount
                elif 1 in poses:
                    expected = poses[1] @ inverse(port_mount) @ inverse(marker_tip) @ mount
            result = estimate_marker(visible[marker_id], marker_sizes[marker_id],
                                     camera, distortion, previous.get(marker_id), expected)
            if result is None or result[1] > 5:
                missing_streak[marker_id] += 1
                if missing_streak[marker_id] > 4:
                    previous.pop(marker_id, None)
                continue
            pose, error, gap, selection = result
            poses[marker_id] = previous[marker_id] = pose
            missing_streak[marker_id] = 0
            markers[str(marker_id)] = {"T_camera_marker": pose.tolist(),
                                       "reprojection_error_px": error,
                                       "pose_ambiguity_gap_px": gap,
                                       "pose_selection": selection}
        record = {"frame_index": index, "timestamp_s": float(timestamp),
                  "visible_ids": sorted(visible), "markers": markers,
                  "quality": "missing_port" if 1 not in poses else "missing_usb",
                  "T_port_usb_marker": None, "usb_pose_source": None,
                  "rgb_file": None}
        if 1 in poses and (0 in poses or (use_second_marker and 10 in poses)):
            # Marker 10 is the primary signal: it remains visible through entry.
            # Marker 0 provides a fallback and an independent agreement check.
            usb_pose = poses[10] @ inverse(mount) if use_second_marker and 10 in poses else poses[0]
            record["usb_pose_source"] = "marker10" if use_second_marker and 10 in poses else "marker0"
            if 0 in poses and 10 in poses:
                record["marker_agreement_m"] = float(np.linalg.norm(
                    poses[0][:3, 3] - usb_pose[:3, 3]))
                record["marker_agreement_rad"] = float((
                    Rotation.from_matrix(poses[0][:3, :3]).inv() *
                    Rotation.from_matrix(usb_pose[:3, :3])).magnitude())
            relative = inverse(poses[1]) @ usb_pose
            xyz, rotvec = translation_rotvec(relative)
            record.update(quality="ok", T_port_usb_marker=relative.tolist(),
                          translation_m=xyz, rotation_vector_rad=rotvec)
        records.append(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = output_dir / "trajectory.jsonl"
    with trajectory.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    metadata = {"source": str(video), "frames": len(records),
                "valid_relative_poses": sum(row["quality"] == "ok" for row in records),
                "mode": "dual" if use_second_marker else "marker0_only",
                "mount_file": str(mount_path), "calibration_file": str(calibration_path),
                "pose_prior_marker_tip": marker_tip.tolist(),
                "image_size_wh": calibration["image_size_wh"]}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_trajectory(records, output_dir / "trajectory.png")
    return metadata
