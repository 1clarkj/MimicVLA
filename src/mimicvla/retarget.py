"""Convert a tracked USB-marker segment into Panda targets and replay it."""

import json
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

from .sim.controller import pose_from_site
from .sim.env import PandaUSBEnv, SimConfig
from .sim.scene import USB_TIP_Z_IN_HAND_M
from .vision.target_frames import provisional_mounts, socket_from_tip


def _matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def _pose_json(pose: np.ndarray) -> list[list[float]]:
    return pose.tolist()


def pose_motion_rates(first: np.ndarray, second: np.ndarray,
                      dt_s: float) -> tuple[float, float]:
    """Translation and rotation rates between two rigid poses."""
    if dt_s <= 0:
        return float("inf"), float("inf")
    linear = float(np.linalg.norm(second[:3, 3] - first[:3, 3]) / dt_s)
    angular = float((Rotation.from_matrix(first[:3, :3]).inv() *
                     Rotation.from_matrix(second[:3, :3])).magnitude() / dt_s)
    return linear, angular


def valid_segment(records: list[dict], start: int | None = None,
                  end: int | None = None) -> list[dict]:
    """Select a contiguous measured segment; never interpolate long marker loss."""
    if start is not None or end is not None:
        selected = records[0 if start is None else start:len(records) if end is None else end + 1]
        if not selected or any(record["quality"] != "ok" for record in selected):
            raise ValueError("Selected frame range must be nonempty and have both markers in every frame")
        return selected
    runs: list[list[dict]] = []
    current: list[dict] = []
    for record in records:
        if record["quality"] == "ok":
            current.append(record)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        raise ValueError("No valid marker pairs")
    return max(runs, key=len)


def _reject_isolated_outliers(positions: np.ndarray, rotations: Rotation) -> np.ndarray:
    keep = np.ones(len(positions), dtype=bool)
    for i in range(1, len(positions) - 1):
        midpoint = (positions[i - 1] + positions[i + 1]) / 2
        neighbor_span = np.linalg.norm(positions[i + 1] - positions[i - 1])
        angular_neighbor = (rotations[i - 1].inv() * rotations[i + 1]).magnitude()
        angular_mid = (rotations[i - 1].inv() * rotations[i]).magnitude()
        if ((np.linalg.norm(positions[i] - midpoint) > .012 and neighbor_span < .012)
                or (angular_mid > np.deg2rad(15) and angular_neighbor < np.deg2rad(8))):
            keep[i] = False
    return keep


def lowpass_poses(positions: np.ndarray, rotations: Rotation, cutoff_hz: float,
                  sample_hz: float) -> tuple[np.ndarray, Rotation]:
    """Zero-phase 2nd-order Butterworth on positions and on small rotation offsets.

    Rotations are filtered as rotation vectors about their mean, which is
    valid for the few-degree wrist motion in these recordings.
    """
    b, a = butter(2, cutoff_hz / (sample_hz / 2))
    if len(positions) <= 3 * max(len(a), len(b)):
        return positions, rotations
    mean = rotations.mean()
    offsets = filtfilt(b, a, (mean.inv() * rotations).as_rotvec(), axis=0)
    return filtfilt(b, a, positions, axis=0), mean * Rotation.from_rotvec(offsets)


def resample_segment(segment: list[dict], control_hz: int = 20,
                     marker_tip: np.ndarray | None = None,
                     lowpass_hz: float | None = None,
                     mirror_socket_x: bool = False) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    """Sample metric socket-relative tip poses at controller rate.

    Gaps are not crossed. A small Gaussian filter reduces lateral jitter;
    axial motion blends back to the unsmoothed signal near the endpoint.
    With ``lowpass_hz``, a zero-phase Butterworth replaces the Gaussian and
    also filters orientation, removing marker-tracking noise above that band.
    With ``mirror_socket_x``, every pose is reflected across the socket's
    local Y-Z plane (p -> M p, R -> M R M with M = diag(-1, 1, 1)). The
    rectangular opening is symmetric about that plane, so the reflected
    motion is an equally valid insertion approached from the other side.
    """
    port_mount, default_marker_tip = provisional_mounts()
    if marker_tip is None:
        marker_tip = default_marker_tip
    raw = []
    for record in segment:
        pose = socket_from_tip(record, port_mount, marker_tip)
        raw.append({"frame_index": record["frame_index"], "timestamp_s": record["timestamp_s"],
                    "T_socket_usb_tip": _pose_json(pose)})
    times = np.asarray([row["timestamp_s"] for row in raw], dtype=float)
    if len(times) < 3 or not np.all(np.diff(times) > 0):
        raise ValueError("Need at least three frames with increasing timestamps")
    poses = np.asarray([row["T_socket_usb_tip"] for row in raw])
    if mirror_socket_x:
        reflect = np.diag([-1., 1., 1., 1.])
        poses = reflect @ poses @ reflect
    positions = poses[:, :3, 3]
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    if any((linear > .75 or angular > 8.0) for linear, angular in
           (pose_motion_rates(poses[i], poses[i + 1], times[i + 1] - times[i])
            for i in range(len(poses) - 1))):
        raise ValueError("Marker pose contains implausible inter-frame motion; review tracking")
    keep = _reject_isolated_outliers(positions, rotations)
    if keep.sum() < 3:
        raise ValueError("Too few poses after outlier rejection")
    # Slerp rejects even a tiny floating-point overshoot beyond the last
    # measured timestamp. Generate interior controller ticks, then append the
    # exact measured endpoint once.
    ticks = np.arange(int(np.ceil((times[-1] - times[0]) * control_hz)))
    interior = times[0] + ticks / control_hz
    sample_times = np.r_[interior[interior < times[-1] - 1e-10], times[-1]]
    sampled_positions = PchipInterpolator(times[keep], positions[keep], axis=0)(sample_times)
    sampled_rotations = Slerp(times[keep], rotations[keep])(sample_times)
    if lowpass_hz is None:
        filtered_positions = gaussian_filter1d(sampled_positions, sigma=.7, axis=0)
    else:
        filtered_positions, sampled_rotations = lowpass_poses(
            sampled_positions, sampled_rotations, lowpass_hz, control_hz)
    # Preserve the final axial motion, including its exact final depth.
    blend = np.clip((sample_times[-1] - sample_times) / (8 / control_hz), 0, 1)
    filtered_positions[:, 2] = (blend * filtered_positions[:, 2] +
                                (1 - blend) * sampled_positions[:, 2])
    filtered_positions[-1] = sampled_positions[-1]
    targets = [_matrix(position, rotation) for position, rotation in
               zip(filtered_positions, sampled_rotations.as_matrix())]
    return sample_times, targets, raw


def _densify(targets: list[np.ndarray], start: np.ndarray,
             max_step_m: float = .003, max_step_rad: float = np.deg2rad(2)) -> list[np.ndarray]:
    """Add controller targets so each command obeys a speed limit."""
    result = []
    previous = start
    for target in targets:
        distance = np.linalg.norm(target[:3, 3] - previous[:3, 3])
        angular = (Rotation.from_matrix(previous[:3, :3]).inv() *
                   Rotation.from_matrix(target[:3, :3])).magnitude()
        count = max(1, int(np.ceil(distance / max_step_m)), int(np.ceil(angular / max_step_rad)))
        interpolator = Slerp([0, 1], Rotation.from_matrix([previous[:3, :3], target[:3, :3]]))
        for fraction in np.arange(1, count + 1) / count:
            position = previous[:3, 3] + fraction * (target[:3, 3] - previous[:3, 3])
            result.append(_matrix(position, interpolator([fraction]).as_matrix()[0]))
        previous = target
    return result


def replay_recording(trajectory: Path, output_dir: Path, *,
                     start_frame: int | None = None, end_frame: int | None = None,
                     complete_segment: bool = False,
                     marker_tip: np.ndarray | None = None,
                     sim_config: SimConfig | None = None,
                     write_video: bool = True,
                     lowpass_hz: float | None = None,
                     mirror_socket_x: bool = False) -> dict:
    records = [json.loads(line) for line in trajectory.read_text().splitlines()]
    segment = valid_segment(records, start_frame, end_frame)
    times, relative_targets, raw = resample_segment(segment, marker_tip=marker_tip,
                                                    lowpass_hz=lowpass_hz,
                                                    mirror_socket_x=mirror_socket_x)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "original_socket_relative.json").write_text(json.dumps(raw, indent=2) + "\n")
    env = PandaUSBEnv(sim_config or SimConfig(max_steps=1000))
    writer = None
    try:
        socket_pos, socket_rot = pose_from_site(env.data, "socket_target")
        world_socket = _matrix(socket_pos, socket_rot)
        ee_tip = _matrix(np.array([0., 0., USB_TIP_Z_IN_HAND_M]), np.eye(3))
        tip_ee = np.linalg.inv(ee_tip)
        world_ee_targets = [world_socket @ relative @ tip_ee for relative in relative_targets]
        current_pos, current_rot = pose_from_site(env.data, "ee_site")
        start_pose = _matrix(current_pos, current_rot)
        initial_alignment_steps = len(_densify(world_ee_targets[:1], start_pose))
        commands = _densify(world_ee_targets, start_pose)
        if len(commands) > env.config.max_steps:
            raise ValueError(f"Replay needs {len(commands)} steps, above episode limit {env.config.max_steps}")
        if write_video:
            writer = imageio.get_writer(output_dir / "replay.mp4", fps=env.config.control_hz,
                                        codec="libx264", quality=7)
        result_rows = []
        for index, target in enumerate(commands):
            delta_position = target[:3, 3] - env.controller.target_position
            delta_rotation = target[:3, :3] @ env.controller.target_rotation.T
            action = np.r_[delta_position, Rotation.from_matrix(delta_rotation).as_rotvec()]
            _, done, info = env.step(action)
            ee_pos, ee_rot = pose_from_site(env.data, "ee_site")
            joint_positions = env.observation(render=False)["observation.state"].tolist()
            joint_ranges = env.model.jnt_range[:7]
            joints = np.asarray(joint_positions[:7])
            within_limits = bool(np.all(joints >= joint_ranges[:, 0] - 1e-4) and
                                 np.all(joints <= joint_ranges[:, 1] + 1e-4))
            result_rows.append({"step": index, "timestamp_s": float(env.data.time),
                                "T_world_ee_target": _pose_json(target),
                                "T_world_ee_actual": _pose_json(_matrix(ee_pos, ee_rot)),
                                "action": action.tolist(), "state": joint_positions,
                                "joint_limits_ok": within_limits, "metrics": info})
            if writer is not None:
                writer.append_data(env.render())
            if done and not complete_segment:
                break
        (output_dir / "retargeted_replay.json").write_text(json.dumps(result_rows, indent=2) + "\n")
        source_rows = result_rows[initial_alignment_steps:]
        first_source_success = next((i for i, row in enumerate(source_rows)
                                     if row["metrics"]["insertion_success"]), None)
        source_prefix = (source_rows[:first_source_success + 1]
                         if first_source_success is not None else [])
        summary = {"source": str(trajectory), "source_frames": [segment[0]["frame_index"],
                    segment[-1]["frame_index"]], "original_frames": len(raw),
                   "insertion_depth_m": env.config.insertion_depth_m,
                   "success_rule": "tip_center_in_opening_v1",
                   "lowpass_hz": lowpass_hz,
                   "mirror_socket_x": mirror_socket_x,
                   "socket_geometry": asdict(env.config.geometry),
                   "T_world_socket": world_socket.tolist(),
                   "T_marker0_usb_tip": (marker_tip if marker_tip is not None
                                         else provisional_mounts()[1]).tolist(),
                   "resampled_targets": len(times), "planned_controller_steps": len(commands),
                   "initial_alignment_steps": initial_alignment_steps,
                   "source_motion_steps": len(source_rows),
                   "source_motion_insertion_reached": first_source_success is not None,
                   "source_steps_to_first_success": first_source_success,
                   "source_prefix_bad_collision_steps": sum(row["metrics"]["bad_collision"]
                                                            for row in source_prefix),
                   "source_prefix_joint_limits_ok": all(row["joint_limits_ok"]
                                                         for row in source_prefix),
                   "controller_steps": len(result_rows),
                   "completed_segment": len(result_rows) == len(commands),
                   "terminated_on_success": len(result_rows) < len(commands) and
                   result_rows[-1]["metrics"]["insertion_success"],
                   "all_joint_limits_ok": all(row["joint_limits_ok"] for row in result_rows),
                   "bad_collision_steps": sum(row["metrics"]["bad_collision"] for row in result_rows),
                   "wall_contact_steps": sum(row["metrics"]["wall_contact"] for row in result_rows),
                   "max_wall_normal_force_n": max(row["metrics"]["max_wall_normal_force_n"] for row in result_rows),
                   "max_wall_penetration_m": max(row["metrics"]["max_wall_penetration_m"] for row in result_rows),
                   "max_axial_depth_m": max(row["metrics"]["axial_error_m"] for row in result_rows),
                   "alignment_reached": any(row["metrics"]["alignment_success"] for row in result_rows),
                   "insertion_reached": any(row["metrics"]["insertion_success"] for row in result_rows),
                   "final_metrics": result_rows[-1]["metrics"]}
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        if writer is not None:
            writer.close()
        env.close()
