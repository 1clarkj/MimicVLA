"""Track printed ArUco markers in handheld-phone video.

T_A_B maps coordinates from frame B to A. Camera C uses OpenCV's optical
coordinates (x right, y down, z forward). Marker frames are centered on the
outer black square, with x right and y up as printed. All distances are metres.
"""

import json
import shutil
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

from .calibrate import load_calibration
from .target_frames import provisional_mounts
from .transforms import inverse, pose_matrix, translation_rotvec


def _select_pose(candidates: list[tuple[float, np.ndarray]],
                 previous: np.ndarray | None,
                 expected: np.ndarray | None = None) -> tuple[float, np.ndarray, str]:
    candidates.sort(key=lambda item: item[0])
    eligible = [item for item in candidates if item[0] <= candidates[0][0] + 1.5]
    if expected is not None and len(eligible) > 1:
        expected_rotation = Rotation.from_matrix(expected[:3, :3])
        deviation = [(expected_rotation.inv() *
                      Rotation.from_matrix(item[1][:3, :3])).magnitude() for item in eligible]
        closest = int(np.argmin(deviation))
        if deviation[closest] < np.deg2rad(35) and max(deviation) > np.deg2rad(70):
            return eligible[closest][0], eligible[closest][1], "task_orientation_prior"
    if previous is None or len(eligible) == 1:
        return candidates[0][0], candidates[0][1], "lowest_reprojection"
    old_rotation = Rotation.from_matrix(previous[:3, :3])
    error, transform = min(eligible, key=lambda item: (
        (old_rotation.inv() * Rotation.from_matrix(item[1][:3, :3])).magnitude()
    ))
    selection = "temporal_continuity" if error > candidates[0][0] + 1e-6 else "lowest_reprojection"
    return error, transform, selection


def estimate_marker(corners: np.ndarray, side_m: float, camera_matrix: np.ndarray,
                    distortion: np.ndarray, previous: np.ndarray | None = None,
                    expected: np.ndarray | None = None
                    ) -> tuple[np.ndarray, float, float, str] | None:
    """Planar-square PnP; prefer pose continuity when reprojection errors are close."""
    half = side_m / 2
    object_points = np.array([[-half, half, 0], [half, half, 0],
                              [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    count, rotations, translations, _ = cv2.solvePnPGeneric(
        object_points, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    candidates = []
    for rvec, tvec in zip(rotations[:count], translations[:count]):
        if float(tvec[2, 0]) <= 0:
            continue
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image_points) ** 2, axis=1))))
        candidates.append((error, pose_matrix(rvec, tvec)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    ambiguity_gap = (candidates[1][0] - candidates[0][0]) if len(candidates) > 1 else None
    # A square seen nearly face-on can have two IPPE poses with similar pixel
    # errors but radically different normals. Motion continuity is more
    # informative than a ~1 px advantage for a single frame.
    error, transform, selection = _select_pose(candidates, previous, expected)
    return transform, error, ambiguity_gap, selection


def detect_pair(frame_bgr: np.ndarray, detector: cv2.aruco.ArucoDetector,
                camera_matrix: np.ndarray, distortion: np.ndarray,
                usb_marker_m: float, port_marker_m: float,
                max_error_px: float = 5.0,
                previous: dict[str, np.ndarray] | None = None,
                annotate: bool = True,
                marker_tip: np.ndarray | None = None) -> tuple[dict, np.ndarray | None]:
    """Return one record and an annotated BGR preview for a single frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    annotated = frame_bgr.copy() if annotate else None
    record = {"visible_ids": [] if ids is None else ids.ravel().astype(int).tolist(),
              "markers": {}, "T_port_usb_marker": None, "quality": "missing_both"}
    duplicate_ids = [] if ids is None else [marker_id for marker_id in (0, 1)
                                            if int(np.count_nonzero(ids == marker_id)) > 1]
    if ids is not None:
        if annotate:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        # Estimate the stationary port first. Its pose gives a weak, explicit
        # orientation prior for disambiguating the planar USB marker.
        for marker_id, side_m, label in [(1, port_marker_m, "port"), (0, usb_marker_m, "usb")]:
            matches = [i for i, value in enumerate(ids.ravel()) if int(value) == marker_id]
            if len(matches) != 1:
                continue  # Duplicate printed IDs are ambiguous.
            expected = None
            if label == "usb" and "port" in record["markers"]:
                port_mount, default_tip = provisional_mounts()
                prior_tip = marker_tip if marker_tip is not None else default_tip
                expected = (np.asarray(record["markers"]["port"]["T_camera_marker"]) @
                            inverse(port_mount) @ inverse(prior_tip))
            result = estimate_marker(corners[matches[0]], side_m, camera_matrix, distortion,
                                     None if previous is None else previous.get(label), expected)
            if result is None:
                continue
            transform, error, ambiguity_gap, selection = result
            record["markers"][label] = {
                "id": marker_id, "reprojection_error_px": error,
                "pose_ambiguity_gap_px": ambiguity_gap,
                "pose_selection": selection,
                "T_camera_marker": transform.tolist(),
            }
            if previous is not None and error <= max_error_px:
                previous[label] = transform
            if annotate:
                rvec, _ = cv2.Rodrigues(transform[:3, :3])
                cv2.drawFrameAxes(annotated, camera_matrix, distortion, rvec,
                                 transform[:3, 3], side_m * 0.5)
    markers = record["markers"]
    if duplicate_ids:
        record["quality"] = "duplicate_id"
    elif len(markers) == 2:
        errors = [markers[k]["reprojection_error_px"] for k in ("usb", "port")]
        if max(errors) <= max_error_px:
            relative = inverse(np.asarray(markers["port"]["T_camera_marker"])) @ np.asarray(
                markers["usb"]["T_camera_marker"]
            )
            xyz, rotvec = translation_rotvec(relative)
            record.update(T_port_usb_marker=relative.tolist(), translation_m=xyz,
                          rotation_vector_rad=rotvec, quality="ok")
        else:
            record["quality"] = "high_reprojection_error"
    elif len(markers) == 1:
        record["quality"] = "missing_port" if "usb" in markers else "missing_usb"
    elif ids is not None and any(int(x) in (0, 1) for x in ids.ravel()):
        record["quality"] = "pose_failed_or_duplicate_id"
    if annotate:
        cv2.putText(annotated, record["quality"], (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 200, 0) if record["quality"] == "ok" else (0, 0, 255), 2)
    return record, annotated


def plot_trajectory(records: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    times = [r["timestamp_s"] for r in records]
    for field, axis, units in [("translation_m", axes[0], "m"),
                               ("rotation_vector_rad", axes[1], "rad")]:
        for coordinate, label in enumerate("xyz"):
            axis.plot(times,
                      [r[field][coordinate] if r["quality"] == "ok" else np.nan
                       for r in records], ".-", label=label)
        axis.set_ylabel(units)
        axis.grid(True)
        axis.legend()
    axes[0].set_title("Blank spans indicate missing or rejected marker poses")
    axes[1].set_xlabel("Video time (s)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def track_video(source: str, calibration_path: Path, output_dir: Path,
                *, usb_marker_m: float, port_marker_m: float = 0.050,
                max_error_px: float = 5.0, max_frames: int | None = None,
                save_frames: bool = True, write_annotated: bool = True,
                marker_tip: np.ndarray | None = None) -> dict:
    if usb_marker_m <= 0 or port_marker_m <= 0:
        raise ValueError("Marker side lengths must be positive")
    calibration = load_calibration(calibration_path)
    capture = cv2.VideoCapture(int(source) if source.isdecimal() else source)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video or camera: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    matrix = np.asarray(calibration["camera_matrix"], dtype=float)
    distortion = np.asarray(calibration["distortion_coefficients"], dtype=float)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(calibration_path, output_dir / "calibration.json")
    frames_dir = output_dir / "frames"
    if save_frames:
        frames_dir.mkdir(exist_ok=True)
    records = []
    previous_poses: dict[str, np.ndarray] = {}
    missing_streak = {"usb": 0, "port": 0}
    writer = None
    started = time.monotonic()
    live = source.isdecimal()
    try:
        while max_frames is None or len(records) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            size = [frame.shape[1], frame.shape[0]]
            if size != calibration["image_size_wh"]:
                raise ValueError(f"Video size {size} differs from calibration "
                                 f"{calibration['image_size_wh']}; use the same iPhone mode")
            index = len(records)
            timestamp = (time.monotonic() - started) if live else (capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            if not live and (not np.isfinite(timestamp) or timestamp <= 0 or
                             (records and timestamp <= records[-1]["timestamp_s"])):
                timestamp = max(index / fps,
                                records[-1]["timestamp_s"] + 1 / fps if records else 0)
            record, annotated = detect_pair(frame, detector, matrix, distortion,
                                            usb_marker_m, port_marker_m, max_error_px,
                                            previous_poses, annotate=write_annotated,
                                            marker_tip=marker_tip)
            for label in ("usb", "port"):
                if (label in record["markers"] and
                        record["markers"][label]["reprojection_error_px"] <= max_error_px):
                    missing_streak[label] = 0
                else:
                    missing_streak[label] += 1
                    if missing_streak[label] > 4:
                        previous_poses.pop(label, None)
            frame_name = f"frame_{index:06d}.jpg"
            if save_frames:
                cv2.imwrite(str(frames_dir / frame_name), frame)
            record.update(frame_index=index, timestamp_s=float(timestamp),
                          rgb_file=f"frames/{frame_name}" if save_frames else None)
            records.append(record)
            if write_annotated and writer is None:
                writer = imageio.get_writer(output_dir / "annotated.mp4", fps=fps,
                                            codec="libx264", quality=7)
            if writer is not None:
                writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    except KeyboardInterrupt:
        if not live:
            raise
    finally:
        capture.release()
        if writer is not None:
            writer.close()
    if not records:
        raise ValueError("No frames decoded from input")
    with (output_dir / "trajectory.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    metadata = {
        "schema": "mimicvla.aruco_trajectory.v1", "source": source,
        "calibration_file": "calibration.json", "image_size_wh": calibration["image_size_wh"],
        "dictionary": "DICT_4X4_50", "usb_marker_id": 0, "port_marker_id": 1,
        "usb_marker_m": usb_marker_m, "port_marker_m": port_marker_m,
        "pose_prior_marker_tip": None if marker_tip is None else marker_tip.tolist(),
        "frames": len(records), "valid_relative_poses": sum(r["quality"] == "ok" for r in records),
        "frame_convention": "T_A_B maps B coordinates into A; OpenCV camera x-right y-down z-forward",
        "timestamps": "video presentation time in seconds; monotonic clock for live camera",
        "saved_frames": save_frames, "annotated_video": write_annotated,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_trajectory(records, output_dir / "trajectory.png")
    return metadata
