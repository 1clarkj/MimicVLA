"""Batch tracking and MuJoCo replay of the recorded insertion videos."""

import csv
import json
import traceback
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation, Slerp

from mimicvla.retarget import pose_motion_rates, replay_recording
from mimicvla.sim.env import SimConfig
from mimicvla.vision.target_frames import provisional_mounts, socket_from_tip
from mimicvla.vision.track import track_video
from mimicvla.vision.transforms import translation_rotvec


def fill_short_gaps(records: list[dict], max_missing: int = 4) -> tuple[list[dict], int]:
    """Interpolate brief relative-pose dropouts; retain original quality labels."""
    filled = [record.copy() for record in records]
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
        if (end - start > max_missing or start == 0 or end == len(records)
                or records[start - 1]["quality"] != "ok"
                or records[end]["quality"] != "ok"):
            continue
        left, right = records[start - 1], records[end]
        t0, t1 = left["timestamp_s"], right["timestamp_s"]
        if t1 <= t0:
            continue
        pose0 = np.asarray(left["T_port_usb_marker"], dtype=float)
        pose1 = np.asarray(right["T_port_usb_marker"], dtype=float)
        translation_speed = np.linalg.norm(pose1[:3, 3] - pose0[:3, 3]) / (t1 - t0)
        rotations = Rotation.from_matrix([pose0[:3, :3], pose1[:3, :3]])
        angular_speed = (rotations[0].inv() * rotations[1]).magnitude() / (t1 - t0)
        if translation_speed > .25 or angular_speed > 2.0:
            continue
        slerp = Slerp([t0, t1], rotations)
        for frame in range(start, end):
            fraction = (records[frame]["timestamp_s"] - t0) / (t1 - t0)
            pose = np.eye(4)
            pose[:3, :3] = slerp([records[frame]["timestamp_s"]]).as_matrix()[0]
            pose[:3, 3] = pose0[:3, 3] + fraction * (pose1[:3, 3] - pose0[:3, 3])
            xyz, rotvec = translation_rotvec(pose)
            filled[frame].update(quality="ok", quality_original=records[frame]["quality"],
                                 interpolated=True, T_port_usb_marker=pose.tolist(),
                                 translation_m=xyz, rotation_vector_rad=rotvec)
            count += 1
    return filled, count


def _consecutive_missing(records: list[dict]):
    current = []
    for record in records:
        if record["quality"] != "ok":
            current.append(record)
        elif current:
            yield current
            current = []
    if current:
        yield current


def _runs(records: list[dict]) -> list[list[dict]]:
    runs = []
    current = []
    for record in records:
        if record["quality"] == "ok":
            if current:
                previous = current[-1]
                linear, angular = pose_motion_rates(
                    np.asarray(previous["T_port_usb_marker"]),
                    np.asarray(record["T_port_usb_marker"]),
                    record["timestamp_s"] - previous["timestamp_s"])
                if linear > .75 or angular > 8.0:
                    runs.append(current)
                    current = []
            current.append(record)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def select_insertion_run(records: list[dict], min_frames: int = 20,
                         marker_tip: np.ndarray | None = None
                         ) -> tuple[list[dict] | None, dict]:
    """Use a fixed, predeclared trajectory-coverage rule for every video."""
    port_mount, default_marker_tip = provisional_mounts()
    if marker_tip is None:
        marker_tip = default_marker_tip
    candidates = []
    for run in _runs(records):
        if len(run) < min_frames:
            continue
        depths = np.array([socket_from_tip(row, port_mount, marker_tip)[2, 3] for row in run])
        depths = median_filter(depths, size=3, mode="nearest")
        start_depth = float(np.median(depths[:3]))
        peak_depth = float(np.max(depths))
        progress = peak_depth - start_depth
        # A long loss of tracking can hide the approach. Keep the continuous
        # insertion after it when that run still shows substantial progress.
        if start_depth <= .005 and peak_depth >= .006 and progress >= .010:
            candidates.append((progress, len(run), run, {
                "start_depth_m": start_depth, "peak_depth_m": peak_depth,
                "axial_progress_m": progress,
                "starts_after_tracking_loss": run[0]["frame_index"] > 0,
                "approach_truncated": start_depth > -.005,
            }))
    if not candidates:
        return None, {"reason": "no_contiguous_20_frame_run_with_10mm_progress_and_6mm_depth"}
    _, _, run, details = max(candidates, key=lambda item: (item[0], item[1]))
    return run, details


def evaluate_recordings(video_dir: Path, calibration: Path, output_dir: Path,
                        *, reuse_existing_replays: bool = False,
                        retrack: bool = False) -> dict:
    """Track each .mov/.mp4, replay one qualifying segment, save CSV and JSON."""
    videos = sorted(path for path in video_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in {".mov", ".mp4"}
                    and path.parent.name.endswith("mm_marker")
                    and path.parent.name.removesuffix("mm_marker").isdigit())
    if not videos:
        raise ValueError(f"No insertion videos found under {video_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for video in videos:
        size_mm = int(video.parent.name.split("mm_marker")[0])
        name = f"{size_mm}mm_{video.stem}"
        folder = output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        row = {"video": str(video), "usb_marker_mm": size_mm,
               "status": "error", "eligible": False, "success": False,
               "contact_free_success": False}
        print(f"Processing {video}", flush=True)
        try:
            tracked = folder / "tracking"
            metadata_path = tracked / "metadata.json"
            if metadata_path.exists() and not retrack:
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("source") != str(video) or metadata.get("usb_marker_m") != size_mm / 1000:
                    raise ValueError("Cached tracking metadata does not match this video and marker size")
            else:
                metadata = track_video(str(video), calibration, tracked,
                                       usb_marker_m=size_mm / 1000,
                                       save_frames=False, write_annotated=False)
            records = [json.loads(line) for line in (tracked / "trajectory.jsonl").read_text().splitlines()]
            row.update(frames=metadata["frames"], valid_poses=metadata["valid_relative_poses"],
                       valid_fraction=metadata["valid_relative_poses"] / metadata["frames"])
            row["longest_raw_tracking_gap_frames"] = max(
                (len(group) for group in _consecutive_missing(records)), default=0)
            row["usb_pose_prior_frames"] = sum(
                record.get("markers", {}).get("usb", {}).get("pose_selection") ==
                "task_orientation_prior" for record in records)
            filled_records, filled_count = fill_short_gaps(records)
            row["interpolated_frames"] = filled_count
            row["implausible_pose_steps"] = sum(
                (linear > .75 or angular > 8.0)
                for first, second in zip(filled_records, filled_records[1:])
                if first["quality"] == second["quality"] == "ok"
                for linear, angular in [pose_motion_rates(
                    np.asarray(first["T_port_usb_marker"]),
                    np.asarray(second["T_port_usb_marker"]),
                    second["timestamp_s"] - first["timestamp_s"])]
            )
            filled_path = tracked / "trajectory_filled.jsonl"
            with filled_path.open("w") as stream:
                for record in filled_records:
                    stream.write(json.dumps(record) + "\n")
            segment, details = select_insertion_run(filled_records)
            row.update(details)
            if segment is None:
                row["status"] = "insufficient_tracking_coverage"
            else:
                row.update(eligible=True, start_frame=segment[0]["frame_index"],
                           end_frame=segment[-1]["frame_index"], segment_frames=len(segment))
                replay_summary = folder / "replay" / "summary.json"
                replay = None
                if reuse_existing_replays and not retrack and replay_summary.exists():
                    cached = json.loads(replay_summary.read_text())
                    if (cached.get("source_frames") ==
                            [segment[0]["frame_index"], segment[-1]["frame_index"]]
                            and cached.get("completed_segment")
                            and cached.get("insertion_depth_m") == SimConfig().insertion_depth_m
                            and cached.get("success_rule") == "tip_center_in_opening_v1"):
                        replay = cached
                if replay is None:
                    replay = replay_recording(filled_path, folder / "replay",
                                              start_frame=segment[0]["frame_index"],
                                              end_frame=segment[-1]["frame_index"],
                                              complete_segment=True)
                row.update(controller_steps=replay["controller_steps"],
                           completed_segment=replay["completed_segment"],
                           insertion_reached=replay["insertion_reached"],
                           bad_collision_steps=replay["bad_collision_steps"],
                           wall_contact_steps=replay["wall_contact_steps"],
                           max_wall_normal_force_n=replay["max_wall_normal_force_n"],
                           max_wall_penetration_m=replay["max_wall_penetration_m"],
                           joint_limits_ok=replay["all_joint_limits_ok"],
                           final_depth_m=replay["final_metrics"]["axial_error_m"],
                           final_radial_error_m=replay["final_metrics"]["radial_error_m"],
                           final_orientation_error_rad=replay["final_metrics"]["orientation_error_rad"])
                row["success"] = bool(replay["insertion_reached"])
                row["replay_quality_ok"] = bool(replay["completed_segment"]
                                                and replay["all_joint_limits_ok"]
                                                and replay["bad_collision_steps"] == 0)
                row["contact_free_success"] = bool(row["success"] and replay["wall_contact_steps"] == 0)
                row["status"] = "success" if row["success"] else "replay_failed"
        except Exception as error:
            row["reason"] = f"{type(error).__name__}: {error}"
            (folder / "error.txt").write_text(traceback.format_exc())
        rows.append(row)
        (output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        fields = sorted({field for result in rows for field in result})
        with (output_dir / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {row['status']}: {row.get('reason', row.get('start_frame', ''))}", flush=True)
    successes = sum(row["success"] for row in rows)
    contact_free = sum(row["contact_free_success"] for row in rows)
    eligible = sum(row["eligible"] for row in rows)
    by_marker_size = {}
    for size_mm in sorted({row["usb_marker_mm"] for row in rows}):
        subset = [row for row in rows if row["usb_marker_mm"] == size_mm]
        by_marker_size[f"{size_mm}mm"] = {
            "videos": len(subset), "eligible": sum(row["eligible"] for row in subset),
            "successes": sum(row["success"] for row in subset),
            "contact_free_successes": sum(row["contact_free_success"] for row in subset),
            "raw_valid_pose_fraction_mean": sum(row.get("valid_fraction", 0) for row in subset) / len(subset),
        }
    summary = {"videos": len(rows), "eligible_for_replay": eligible,
               "insertion_depth_m": SimConfig().insertion_depth_m,
               "replay_successes": successes,
               "contact_free_successes": contact_free,
               "end_to_end_success_rate": successes / len(rows),
               "contact_free_success_rate": contact_free / len(rows),
               "success_rate_among_eligible": successes / eligible if eligible else None,
               "by_marker_size": by_marker_size,
               "gap_rule": "interpolate at most 4 missing frames if adjacent relative poses imply <=0.25 m/s translation and <=2 rad/s rotation; preserve raw trajectory",
               "selection_rule": "contiguous >=20 valid frames; starts no deeper than 5mm inside socket; reaches >=6mm depth; >=10mm axial progress; choose greatest progress; flag truncated approach",
               "success_rule": "tip centre inside modeled rectangular port opening at 12-18mm depth; orientation and contact diagnostic only",
               "contact_free_rule": "geometric success and no physical plug/front-wall contact at any replay step"}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
