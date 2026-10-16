"""Rescore recorded MuJoCo steps when the physical insertion depth changes."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..sim.env import SimConfig, insertion_success_from_metrics
from ..sim.scene import (SOCKET_DEPTH_M, SOCKET_INNER_HALF_X_M,
                         SOCKET_INNER_HALF_Y_M, USB_TIP_Z_IN_HAND_M, SocketPose)


DEFAULT_BATCHES = {
    "single_marker_original": Path("outputs/video_evaluation_pose_prior"),
    "dual_marker_offset_minus3mm": Path("outputs/dual_marker_comparison_tip_xminus3"),
    "dual_marker_original_mount": Path("outputs/dual_marker_original_mount"),
}


def _replay_folder(root: Path, row: dict) -> Path:
    stem = Path(row["video"]).stem
    if "mode" in row:
        return root / row["mode"] / stem / "replay"
    return root / f"{int(row['usb_marker_mm'])}mm_{stem}" / "replay"


def _metrics_with_lateral_position(step: dict, socket_pose: np.ndarray) -> dict:
    """Recover tip-in-socket coordinates from saved actual EE pose in old replays."""
    metrics = step["metrics"].copy()
    if "socket_lateral_x_m" in metrics and "socket_lateral_y_m" in metrics:
        return metrics
    ee = np.asarray(step["T_world_ee_actual"], dtype=float)
    tip_world = ee[:3, 3] + ee[:3, :3] @ np.array([0.0, 0.0, USB_TIP_Z_IN_HAND_M])
    local = socket_pose[:3, :3].T @ (tip_world - socket_pose[:3, 3])
    metrics["socket_lateral_x_m"] = float(local[0])
    metrics["socket_lateral_y_m"] = float(local[1])
    return metrics


def rescore_replays(output_dir: Path, batches: dict[str, Path] | None = None,
                   required_depth_m: float | None = None) -> dict:
    """Use saved actual states/contact metrics; preserve legacy reports."""
    batches = DEFAULT_BATCHES if batches is None else batches
    required_depth_m = (SimConfig().insertion_depth_m if required_depth_m is None
                        else required_depth_m)
    if not 0 < required_depth_m <= SOCKET_DEPTH_M:
        raise ValueError("Required insertion depth must fit within the modeled socket")
    results = []
    for batch, root in batches.items():
        for source_row in json.loads((root / "results.json").read_text()):
            row = {"batch": batch, "mode": source_row.get("mode", "single_marker"),
                   "video": source_row["video"], "replay_eligible": bool(source_row["eligible"]),
                   "depth_reached": False, "geometry_reached": False,
                   "insertion_reached": False, "episode_success": False,
                   "contact_free_success": False}
            if source_row["eligible"]:
                replay = _replay_folder(root, source_row)
                steps = json.loads((replay / "retargeted_replay.json").read_text())
                previous = json.loads((replay / "summary.json").read_text())
                socket_pose = previous.get("T_world_socket")
                if socket_pose is None:
                    socket = SocketPose()  # Historical replay_recording used this fixed pose.
                    quaternion = np.asarray(socket.quaternion_wxyz)
                    socket_pose = np.eye(4)
                    socket_pose[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
                    socket_pose[:3, 3] = socket.position_m
                socket_pose = np.asarray(socket_pose, dtype=float)
                metrics = [_metrics_with_lateral_position(step, socket_pose) for step in steps]
                row["max_axial_depth_mm"] = max(m["axial_error_m"] for m in metrics) * 1000
                row["depth_reached"] = any(m["axial_error_m"] >= required_depth_m for m in metrics)
                row["insertion_reached"] = any(insertion_success_from_metrics(
                    m, required_depth_m=required_depth_m) for m in metrics)
                row["geometry_reached"] = row["insertion_reached"]
                row["bad_collision_steps"] = previous["bad_collision_steps"]
                row["wall_contact_steps"] = previous["wall_contact_steps"]
                row["completed_segment"] = previous["completed_segment"]
                row["joint_limits_ok"] = previous["all_joint_limits_ok"]
                row["episode_success"] = bool(row["insertion_reached"])
                row["replay_quality_ok"] = bool(row["completed_segment"]
                                                 and row["joint_limits_ok"]
                                                 and row["bad_collision_steps"] == 0)
                row["contact_free_success"] = bool(row["episode_success"]
                                                   and row["wall_contact_steps"] == 0)
            results.append(row)
    grouped = defaultdict(list)
    for row in results:
        grouped[(row["batch"], row["mode"])].append(row)
    by_batch = {f"{batch}/{mode}": {
        "videos": len(group),
        "replay_eligible": sum(row["replay_eligible"] for row in group),
        "reached_required_depth": sum(row["depth_reached"] for row in group),
        "geometry_reached": sum(row["geometry_reached"] for row in group),
        "episode_successes": sum(row["episode_success"] for row in group),
        "contact_free_successes": sum(row["contact_free_success"] for row in group),
    } for (batch, mode), group in grouped.items()}
    summary = {"required_tip_depth_mm": required_depth_m * 1000,
               "socket_model_depth_mm": SOCKET_DEPTH_M * 1000,
               "socket_opening_half_width_mm": SOCKET_INNER_HALF_X_M * 1000,
               "socket_opening_half_height_mm": SOCKET_INNER_HALF_Y_M * 1000,
               "success_rule": "tip centre inside rectangular opening at >= required depth and <= modeled socket depth",
               "ordinary_wall_contact_allowed": True,
               "replay_quality_rule": "completed segment, valid joints, no >1 mm socket penetration; diagnostic only",
               "contact_free_rule": "diagnostic only: episode success and zero front-wall contact steps",
               "source": "saved actual MuJoCo step metrics; source replays were not rerun",
               "by_batch": by_batch}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    fields = sorted({key for row in results for key in row})
    with (output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    return summary
