"""Track the plug and port markers in a handheld-phone video.

T_A_B maps coordinates from frame B to A. Camera C uses OpenCV's optical
coordinates (x right, y down, z forward). Marker frames are centered on the
outer black square, with x right and y up as printed. All distances are metres.
Every marker id, size and transform comes from a setup file (vision.setup).
"""

import json
import shutil
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .calibrate import load_calibration
from .setup import Setup
from .transforms import inverse, pose_matrix


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


def plot_trajectory(records: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    times = [r["timestamp_s"] for r in records]
    poses = [np.asarray(r["T_socket_usb_tip"]) if r["T_socket_usb_tip"] else None
             for r in records]
    series = [("USB tip in socket frame (mm)", lambda pose: 1000 * pose[:3, 3]),
              ("rotation vector (rad)",
               lambda pose: Rotation.from_matrix(pose[:3, :3]).as_rotvec())]
    for axis, (units, value) in zip(axes, series):
        values = np.array([value(pose) if pose is not None else [np.nan] * 3
                           for pose in poses])
        for coordinate, label in enumerate("xyz"):
            axis.plot(times, values[:, coordinate], ".-", label=label)
        axis.set_ylabel(units)
        axis.grid(True)
        axis.legend()
    axes[0].set_title("Blank spans are frames without a pose; z > 0 is inside the port")
    axes[1].set_xlabel("Video time (s)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def fill_short_gaps(records: list[dict], max_missing: int = 4) -> int:
    """Interpolate dropouts of at most ``max_missing`` frames in place.

    Only gaps whose neighbouring poses imply at most 0.25 m/s and 2 rad/s are
    filled; longer or faster gaps stay missing. Returns the filled count.
    """
    count = 0
    index = 0
    while index < len(records):
        if records[index]["quality"] == "ok":
            index += 1
            continue
        start = index
        while index < len(records) and records[index]["quality"] != "ok":
            index += 1
        end = index
        if end - start > max_missing or start == 0 or end == len(records):
            continue
        left, right = records[start - 1], records[end]
        t0, t1 = left["timestamp_s"], right["timestamp_s"]
        if t1 <= t0:
            continue
        pose0 = np.asarray(left["T_port_usb_marker"], dtype=float)
        pose1 = np.asarray(right["T_port_usb_marker"], dtype=float)
        rotations = Rotation.from_matrix([pose0[:3, :3], pose1[:3, :3]])
        if (np.linalg.norm(pose1[:3, 3] - pose0[:3, 3]) / (t1 - t0) > .25 or
                (rotations[0].inv() * rotations[1]).magnitude() / (t1 - t0) > 2.0):
            continue
        slerp = Slerp([t0, t1], rotations)
        for frame in range(start, end):
            fraction = (records[frame]["timestamp_s"] - t0) / (t1 - t0)
            pose = np.eye(4)
            pose[:3, :3] = slerp([records[frame]["timestamp_s"]]).as_matrix()[0]
            pose[:3, 3] = pose0[:3, 3] + fraction * (pose1[:3, 3] - pose0[:3, 3])
            records[frame].update(quality="ok", quality_original=records[frame]["quality"],
                                  pose_source="interpolated", T_port_usb_marker=pose.tolist())
            count += 1
    return count


def _frames(video: Path, max_frames: int | None = None):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    previous = None
    try:
        index = 0
        while max_frames is None or index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not np.isfinite(timestamp) or timestamp <= 0 or (
                    previous is not None and timestamp <= previous):
                timestamp = max(index / fps, previous + 1 / fps if previous is not None else 0)
            previous = timestamp
            yield index, timestamp, fps, frame
            index += 1
    finally:
        capture.release()


def _visible(frame: np.ndarray, detector) -> dict[int, np.ndarray]:
    """Detected marker corners by id; ids seen more than once are ambiguous and dropped."""
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return {}
    ids = ids.ravel().astype(int).tolist()
    return {marker_id: corner for marker_id, corner in zip(ids, corners)
            if ids.count(marker_id) == 1}


def _detector():
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def _draw_point(frame, transform, camera, distortion, colour, text, scale) -> None:
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    cv2.drawFrameAxes(frame, camera, distortion, rvec, transform[:3, 3], .012, 3)
    pixels, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, transform[:3, 3], camera, distortion)
    x, y = np.rint(pixels[0, 0]).astype(int)
    cv2.circle(frame, (x, y), int(10 * scale), colour, 3)
    cv2.putText(frame, text, (x + int(14 * scale), y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, 3)


def track_video(video: Path, setup: Setup, output_dir: Path, *, max_error_px: float = 5.0,
                max_frames: int | None = None, annotate: bool = True) -> dict:
    """Write trajectory.jsonl with T_port_usb_marker and T_socket_usb_tip per frame.

    The port marker is estimated first; its pose and the setup transforms give
    an orientation prior that resolves the planar-marker ambiguity of the plug
    marker. With a second plug marker, the second marker is the primary signal
    (it stays visible during entry) and the first is the fallback. Gaps of up
    to four frames are interpolated; longer ones stay missing. ``annotate``
    writes annotated.mp4 with the marker axes and the projected socket and
    plug tip, which is the quickest check of a new setup's transforms.
    """
    calibration = load_calibration(setup.camera)
    camera = np.asarray(calibration["camera_matrix"], dtype=float)
    distortion = np.asarray(calibration["distortion_coefficients"], dtype=float)
    port, usb, usb2 = setup.port_marker, setup.usb_marker, setup.usb_marker_2
    if usb2 is not None and setup.T_usb_marker_usb_marker_2 is None:
        raise ValueError("The setup has a second plug marker without its transform; "
                         "run `mimicvla fit-marker-mount` first")
    mount = setup.T_usb_marker_usb_marker_2
    socket_port = setup.T_socket_port_marker
    marker_tip = setup.T_usb_marker_usb_tip
    detector = _detector()
    output_dir.mkdir(parents=True, exist_ok=True)
    previous: dict[int, np.ndarray] = {}
    missing = {marker.id: 0 for marker in (port, usb, usb2) if marker is not None}
    records = []
    writer = None
    try:
        for index, timestamp, fps, frame in _frames(video, max_frames):
            if [frame.shape[1], frame.shape[0]] != calibration["image_size_wh"]:
                raise ValueError(f"Video size {[frame.shape[1], frame.shape[0]]} differs from "
                                 f"the calibration {calibration['image_size_wh']}; use the "
                                 "same camera mode as for the calibration")
            visible = _visible(frame, detector)
            poses, markers = {}, {}
            for marker in (port, usb, usb2):
                if marker is None:
                    continue
                expected = None
                if marker is usb and port.id in poses:
                    expected = poses[port.id] @ inverse(socket_port) @ inverse(marker_tip)
                elif marker is usb2:
                    if usb.id in poses:
                        expected = poses[usb.id] @ mount
                    elif port.id in poses:
                        expected = (poses[port.id] @ inverse(socket_port) @
                                    inverse(marker_tip) @ mount)
                result = (estimate_marker(visible[marker.id], marker.size_m, camera, distortion,
                                          previous.get(marker.id), expected)
                          if marker.id in visible else None)
                if result is None or result[1] > max_error_px:
                    missing[marker.id] += 1
                    if missing[marker.id] > 4:
                        previous.pop(marker.id, None)
                    continue
                pose, error, gap, selection = result
                poses[marker.id] = previous[marker.id] = pose
                missing[marker.id] = 0
                markers[str(marker.id)] = {"T_camera_marker": pose.tolist(),
                                           "reprojection_error_px": error,
                                           "pose_ambiguity_gap_px": gap,
                                           "pose_selection": selection}
            record = {"frame_index": index, "timestamp_s": float(timestamp),
                      "visible_ids": sorted(visible), "markers": markers,
                      "quality": "missing_port" if port.id not in poses else "missing_usb",
                      "pose_source": "missing", "usb_pose_source": None,
                      "T_port_usb_marker": None, "T_socket_usb_tip": None}
            usb_pose = None
            if usb2 is not None and usb2.id in poses:
                usb_pose, record["usb_pose_source"] = poses[usb2.id] @ inverse(mount), "usb_marker_2"
            elif usb.id in poses:
                usb_pose, record["usb_pose_source"] = poses[usb.id], "usb_marker"
            if port.id in poses and usb_pose is not None:
                record.update(quality="ok", pose_source="observed",
                              T_port_usb_marker=(inverse(poses[port.id]) @ usb_pose).tolist())
            records.append(record)
            if annotate:
                scale = frame.shape[1] / 1280
                if visible:
                    cv2.aruco.drawDetectedMarkers(frame, list(visible.values()),
                                                  np.array(list(visible)).reshape(-1, 1))
                if port.id in poses:
                    _draw_point(frame, poses[port.id] @ inverse(socket_port), camera,
                                distortion, (0, 255, 0), "socket", scale)
                if usb_pose is not None:
                    _draw_point(frame, usb_pose @ marker_tip, camera, distortion,
                                (255, 0, 255), "plug tip", scale)
                small = cv2.resize(frame, (1280, int(round(frame.shape[0] / scale / 2)) * 2),
                                   interpolation=cv2.INTER_AREA)
                if writer is None:
                    writer = imageio.get_writer(output_dir / "annotated.mp4", fps=fps,
                                                codec="libx264", quality=7)
                writer.append_data(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    finally:
        if writer is not None:
            writer.close()
    if not records:
        raise ValueError(f"No frames decoded from {video}")
    filled = fill_short_gaps(records)
    for record in records:
        if record["quality"] == "ok":
            record["T_socket_usb_tip"] = setup.socket_from_usb_marker(
                np.asarray(record["T_port_usb_marker"])).tolist()
    with (output_dir / "trajectory.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    if setup.path is not None:
        shutil.copyfile(setup.path, output_dir / "setup.json")
    metadata = {"source": str(video), "setup": str(setup.path), "frames": len(records),
                "observed_poses": sum(r["pose_source"] == "observed" for r in records),
                "interpolated_poses": filled,
                "image_size_wh": calibration["image_size_wh"]}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_trajectory(records, output_dir / "trajectory.png")
    return metadata


def fit_marker_mount(videos: list[Path], setup: Setup) -> dict:
    """Fit T_usb_marker_usb_marker_2 from frames where both plug markers are seen."""
    calibration = load_calibration(setup.camera)
    camera = np.asarray(calibration["camera_matrix"])
    distortion = np.asarray(calibration["distortion_coefficients"])
    first, second = setup.usb_marker, setup.usb_marker_2
    if second is None:
        raise ValueError("The setup has no usb_marker_2")
    detector = _detector()
    transforms = []
    for video in videos:
        previous = {}
        for _, _, _, frame in _frames(video):
            visible = _visible(frame, detector)
            poses = {}
            for marker in (first, second):
                if marker.id not in visible:
                    continue
                estimate = estimate_marker(visible[marker.id], marker.size_m, camera,
                                           distortion, previous.get(marker.id))
                if estimate is not None and estimate[1] <= 5:
                    poses[marker.id] = previous[marker.id] = estimate[0]
            if len(poses) == 2:
                transforms.append(inverse(poses[first.id]) @ poses[second.id])
    if len(transforms) < 20:
        raise ValueError(f"Need at least 20 frames with both plug markers; got {len(transforms)}")
    transforms = np.asarray(transforms)
    positions = transforms[:, :3, 3]
    rotations = Rotation.from_matrix(transforms[:, :3, :3])
    keep = ((np.linalg.norm(positions - np.median(positions, axis=0), axis=1) < .015) &
            ((rotations.mean().inv() * rotations).magnitude() < np.deg2rad(15)))
    if keep.sum() < 20:
        raise ValueError("The two marker poses do not support one rigid mount")
    mount = np.eye(4)
    mount[:3, :3] = rotations[keep].mean().as_matrix()
    mount[:3, 3] = np.median(positions[keep], axis=0)
    residual = np.linalg.norm(positions[keep] - mount[:3, 3], axis=1)
    return {"T_usb_marker_usb_marker_2": mount.tolist(), "frames": len(transforms),
            "inlier_frames": int(keep.sum()),
            "translation_residual_mm_p50": float(np.median(residual) * 1000)}
