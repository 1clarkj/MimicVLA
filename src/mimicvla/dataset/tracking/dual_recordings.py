"""Compare marker-0-only and two-marker replay on the same new videos."""

import csv
import json
import shutil
import traceback
from pathlib import Path

import numpy as np

from mimicvla.dataset.tracking.recordings import _consecutive_missing, fill_short_gaps, select_insertion_run
from mimicvla.retarget import pose_motion_rates, replay_recording
from mimicvla.vision.dual_marker import estimate_mount, track_dual_video
from mimicvla.vision.target_frames import marker_tip_with_x_offset


KNOWN_BATCHES = {
    "double_marker": (-3.0, Path("outputs/dual_marker_comparison_tip_xminus3")),
    "double_marker_original_mount": (0.0, Path("outputs/dual_marker_original_mount")),
}


def resolve_batch(video_dir: Path, tip_x_mm: float | None,
                  output_dir: Path | None) -> tuple[float, Path]:
    """Keep recordings from different physical USB mounts separate."""
    defaults = KNOWN_BATCHES.get(video_dir.name)
    if tip_x_mm is None:
        if defaults is None:
            raise ValueError("Unknown recording batch: pass --tip-x-mm and --output-dir")
        tip_x_mm = defaults[0]
    if output_dir is None:
        if defaults is None or tip_x_mm != defaults[0]:
            raise ValueError("Non-default mount: pass a separate --output-dir")
        output_dir = defaults[1]
    return tip_x_mm, output_dir


def evaluate_dual_recordings(video_dir: Path, calibration: Path,
                             output_dir: Path | None = None, *, marker10_mm: float = 22.0,
                             tip_x_mm: float | None = None,
                             reuse_tracking_from: Path | None = None) -> dict:
    tip_x_mm, output_dir = resolve_batch(video_dir, tip_x_mm, output_dir)
    videos = sorted(video_dir.glob("*.mov")) + sorted(video_dir.glob("*.mp4"))
    if not videos:
        raise ValueError(f"No videos under {video_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    mount_path = output_dir / "usb_dual_mount.json"
    if reuse_tracking_from is None:
        mount = estimate_mount(videos, calibration, mount_path,
                               marker10_m=marker10_mm / 1000)
    else:
        mount = json.loads((reuse_tracking_from / "usb_dual_mount.json").read_text())
        if mount_path.resolve() != (reuse_tracking_from / "usb_dual_mount.json").resolve():
            shutil.copyfile(reuse_tracking_from / "usb_dual_mount.json", mount_path)
    marker_tip = marker_tip_with_x_offset(tip_x_mm)
    rows = []
    for mode, use_second in (("marker0_only", False), ("dual", True)):
        for video in videos:
            folder = output_dir / mode / video.stem
            row = {"mode": mode, "video": str(video), "status": "error",
                   "eligible": False, "success": False, "contact_free_success": False}
            print(f"{mode}: {video.name}", flush=True)
            try:
                tracked = folder / "tracking"
                source_tracked = (reuse_tracking_from / mode / video.stem / "tracking"
                                  if reuse_tracking_from is not None else tracked)
                if reuse_tracking_from is None:
                    metadata = track_dual_video(video, calibration, tracked, mount_path,
                                                use_second_marker=use_second,
                                                marker10_m=marker10_mm / 1000)
                else:
                    metadata = json.loads((source_tracked / "metadata.json").read_text())
                    if metadata["source"] != str(video):
                        raise ValueError("Cached tracking belongs to a different video")
                records = [json.loads(line) for line in
                           (source_tracked / "trajectory.jsonl").read_text().splitlines()]
                row.update(frames=metadata["frames"], valid_poses=metadata["valid_relative_poses"],
                           valid_fraction=metadata["valid_relative_poses"] / metadata["frames"],
                           longest_raw_tracking_gap_frames=max(
                               (len(group) for group in _consecutive_missing(records)), default=0),
                           marker10_frames=sum(r["usb_pose_source"] == "marker10" for r in records),
                           marker0_fallback_frames=sum(r["usb_pose_source"] == "marker0" for r in records))
                agreement = [r["marker_agreement_m"] for r in records
                             if "marker_agreement_m" in r]
                row["marker_agreement_mm_p50"] = (float(np.median(agreement) * 1000)
                                                   if agreement else None)
                row["marker_agreement_mm_p95"] = (float(np.percentile(agreement, 95) * 1000)
                                                   if agreement else None)
                row["marker_disagreement_over_5mm_frames"] = sum(
                    value > .005 for value in agreement)
                filled, count = fill_short_gaps(records)
                row["interpolated_frames"] = count
                row["implausible_pose_steps"] = sum(
                    linear > .75 or angular > 8.0
                    for a, b in zip(filled, filled[1:])
                    if a["quality"] == b["quality"] == "ok"
                    for linear, angular in [pose_motion_rates(
                        np.asarray(a["T_port_usb_marker"]),
                        np.asarray(b["T_port_usb_marker"]),
                        b["timestamp_s"] - a["timestamp_s"])]
                )
                filled_path = tracked / "trajectory_filled.jsonl"
                tracked.mkdir(parents=True, exist_ok=True)
                with filled_path.open("w") as stream:
                    for record in filled:
                        stream.write(json.dumps(record) + "\n")
                segment, details = select_insertion_run(filled, marker_tip=marker_tip)
                row.update(details)
                if segment is None:
                    row["status"] = "insufficient_tracking_coverage"
                else:
                    row.update(eligible=True, start_frame=segment[0]["frame_index"],
                               end_frame=segment[-1]["frame_index"],
                               segment_frames=len(segment))
                    replay = replay_recording(filled_path, folder / "replay",
                                              start_frame=segment[0]["frame_index"],
                                              end_frame=segment[-1]["frame_index"],
                                              complete_segment=True,
                                              marker_tip=marker_tip)
                    row.update(controller_steps=replay["controller_steps"],
                               insertion_reached=replay["insertion_reached"],
                               alignment_reached=replay["alignment_reached"],
                               max_axial_depth_mm=replay["max_axial_depth_m"] * 1000,
                               completed_segment=replay["completed_segment"],
                               joint_limits_ok=replay["all_joint_limits_ok"],
                               bad_collision_steps=replay["bad_collision_steps"],
                               wall_contact_steps=replay["wall_contact_steps"],
                               max_wall_penetration_mm=replay["max_wall_penetration_m"] * 1000,
                               max_wall_normal_force_n=replay["max_wall_normal_force_n"])
                    row["success"] = bool(replay["insertion_reached"])
                    row["replay_quality_ok"] = bool(replay["completed_segment"]
                                                    and replay["all_joint_limits_ok"]
                                                    and replay["bad_collision_steps"] == 0)
                    row["contact_free_success"] = bool(row["success"] and
                                                       replay["wall_contact_steps"] == 0)
                    row["status"] = "success" if row["success"] else "replay_failed"
            except Exception as error:
                row["reason"] = f"{type(error).__name__}: {error}"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "error.txt").write_text(traceback.format_exc())
            rows.append(row)
            print(f"  {row['status']}: {row.get('reason', row.get('start_frame', ''))}", flush=True)
            (output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
            fields = sorted({key for item in rows for key in item})
            with (output_dir / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    by_mode = {}
    for mode in ("marker0_only", "dual"):
        subset = [row for row in rows if row["mode"] == mode]
        by_mode[mode] = {"videos": len(subset),
                         "valid_frames": sum(row.get("valid_poses", 0) for row in subset),
                         "eligible": sum(row["eligible"] for row in subset),
                         "alignment_reached": sum(row.get("alignment_reached", False) for row in subset),
                         "reached_6mm_depth": sum(row.get("max_axial_depth_mm", 0) >= 6 for row in subset),
                         "geometric_successes": sum(row["success"] for row in subset),
                         "contact_free_successes": sum(row["contact_free_success"] for row in subset),
                         "errors": sum(row["status"] == "error" for row in subset)}
    summary = {"mount": str(mount_path), "mount_shared_frames": mount["shared_frames"],
               "recording_batch": video_dir.name,
               "tip_x_mm": tip_x_mm,
               "marker_tip_transform": marker_tip.tolist(),
               "tip_offset_interpretation": ("The marker centre is +3 mm along X from the tip centre "
                                             "for the new mount; therefore T_marker0_tip has X=-3 mm"
                                             if tip_x_mm == -3.0 else "X is tip centre relative to marker 0"),
               "tracking_reused_from": str(reuse_tracking_from) if reuse_tracking_from else None,
               "by_mode": by_mode,
               "note": (f"Same {len(videos)} source videos and replay rule; marker 10 side={marker10_mm} mm, "
                        f"marker-0-to-tip X={tip_x_mm} mm. This is trajectory replay, "
                        "not learned-policy evaluation.")}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
