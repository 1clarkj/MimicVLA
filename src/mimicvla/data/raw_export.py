"""Self-contained VideoFolder-style archive of human videos and tracked poses."""

import hashlib
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import av
import imageio_ffmpeg

from ..vision.target_frames import provisional_mounts, socket_from_tip


INITIAL_RECORDING_IDS = frozenset({
    "12mm_10102026_155357", "12mm_10102026_155414",
    "12mm_10102026_155430", "12mm_10102026_155442",
    "20mm_10102026_161008", "20mm_10102026_161205",
    "20mm_10102026_161220", "20mm_10102026_161242",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_video_only(path: Path, tracked_times: list[float]) -> int:
    """Check audio removal and alignment of every tracked frame's timestamp."""
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio):
            raise ValueError(f"Expected one video stream and no audio: {path}")
        frame_times = [float(frame.time) for frame in container.decode(video=0)]
    if not len(tracked_times) <= len(frame_times) <= len(tracked_times) + 2:
        raise ValueError(f"Video/trajectory frame count mismatch: {path}: "
                         f"{len(frame_times)} video vs {len(tracked_times)} tracked")
    if any(abs(video_time - pose_time) > 0.015 for video_time, pose_time in
           zip(frame_times, tracked_times)):
        raise ValueError(f"Video/trajectory timestamps do not align: {path}")
    return len(frame_times)


def _export_video_only(source: Path, target: Path, tracked_times: list[float]) -> int:
    """Copy the original video stream without audio; never rewrite the source."""
    if target.exists():
        try:
            return _check_video_only(target, tracked_times)
        except ValueError:
            pass  # Older exports are source hardlinks with audio.
    temporary = target.with_name(target.stem + ".silent_tmp" + target.suffix)
    try:
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-nostdin", "-y", "-i", str(source), "-map", "0:v:0", "-c:v", "copy",
            "-an", "-sn", "-dn", "-movflags", "+faststart", str(temporary),
        ], check=True)
        decoded_frames = _check_video_only(temporary, tracked_times)
        os.replace(temporary, target)
        return decoded_frames
    finally:
        temporary.unlink(missing_ok=True)


def _policy_screen(row: dict) -> list[str]:
    """Conservative exclusion reasons, separate from archival membership."""
    if not row["eligible"]:
        return ["incomplete_tracked_insertion_span"]
    reasons = []
    if not row.get("completed_segment", False):
        reasons.append("incomplete_robot_replay")
    if not row.get("joint_limits_ok", False):
        reasons.append("joint_limit_violation")
    if row.get("bad_collision_steps", 0):
        reasons.append("deep_collision")
    if not row.get("insertion_reached", False):
        reasons.append("no_geometric_insertion")
    if row.get("wall_contact_steps", 0):
        reasons.append("wall_contact")
    return reasons


def export_raw_dataset(evaluation_dir: Path, calibration: Path,
                       output_dir: Path) -> dict:
    """Export all videos, raw tracked poses, and physical frame metadata.

    This is intentionally not a LeRobotDataset: human videos have no Panda
    observations or actions. The videos/metadata.jsonl follows VideoFolder.
    """
    rows = json.loads((evaluation_dir / "results.json").read_text())
    video_dir = output_dir / "videos"
    prior_metadata = video_dir / "metadata.jsonl"
    previous_ids = ({json.loads(line)["episode_id"] for line in prior_metadata.read_text().splitlines()}
                    if prior_metadata.exists() else set())
    pose_dir = output_dir / "poses"
    processed_dir = output_dir / "poses_processed"
    calibration_dir = output_dir / "calibration"
    for folder in (video_dir, pose_dir, processed_dir, calibration_dir):
        folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(calibration, calibration_dir / "iphone_calibration.json")
    for pdf in sorted(Path("calibration").glob("*.pdf")):
        shutil.copy2(pdf, calibration_dir / pdf.name)
    port_mount, marker_tip = provisional_mounts()
    transforms = {
        "convention": "T_A_B maps coordinates in B into A; translations in metres",
        "T_socket_port_marker": port_mount.tolist(),
        "T_usb_marker_usb_tip": marker_tip.tolist(),
        "port_marker_position_in_socket_mm": [16, -45, 0],
        "usb_tip_position_in_usb_marker_mm": [0, 30, -3],
        "socket_plus_z": "inward through laptop wall",
        "marker_plus_z": "outward from printed face",
        "orientation_note": "Axis alignment inferred from marker normals and near-insertion video; target points visually confirmed by user",
    }
    (calibration_dir / "marker_mounts.json").write_text(json.dumps(transforms, indent=2) + "\n")
    metadata = []
    for row in rows:
        source = Path(row["video"])
        size_mm = row["usb_marker_mm"]
        episode_id = f"{size_mm}mm_{source.stem}"
        copied_video = video_dir / f"{episode_id}{source.suffix.lower()}"
        tracked = evaluation_dir / episode_id / "tracking" / "trajectory.jsonl"
        if not tracked.exists():
            raise FileNotFoundError(f"Missing raw tracking for {source}; run evaluate-recordings first")
        raw_records = [json.loads(line) for line in tracked.read_text().splitlines()]
        video_frames = _export_video_only(source, copied_video,
                                          [record["timestamp_s"] for record in raw_records])
        frame_count = 0
        valid_count = 0
        processed_valid_count = 0
        poses_file = pose_dir / f"{episode_id}.jsonl"
        with poses_file.open("w") as target_stream:
            for source_record in raw_records:
                record = source_record.copy()
                record.pop("rgb_file", None)  # Frames are indexed within the original video.
                pose = socket_from_tip(record, port_mount, marker_tip)
                record["T_socket_usb_tip"] = None if pose is None else pose.tolist()
                record["video_file"] = f"../videos/{copied_video.name}"
                target_stream.write(json.dumps(record) + "\n")
                frame_count += 1
                valid_count += pose is not None
        if frame_count != row["frames"] or valid_count != row["valid_poses"]:
            raise ValueError(f"Tracking count mismatch for {source}")
        filled_path = evaluation_dir / episode_id / "tracking" / "trajectory_filled.jsonl"
        if not filled_path.exists():
            raise FileNotFoundError(f"Missing preprocessed tracking for {source}; run evaluate-recordings first")
        processed_file = processed_dir / f"{episode_id}.jsonl"
        with filled_path.open() as source_stream, processed_file.open("w") as target_stream:
            for line in source_stream:
                record = json.loads(line)
                record.pop("rgb_file", None)
                pose = socket_from_tip(record, port_mount, marker_tip)
                record["T_socket_usb_tip"] = None if pose is None else pose.tolist()
                record["video_file"] = f"../videos/{copied_video.name}"
                record["pose_source"] = ("interpolated" if record.get("interpolated") else
                                         "observed" if pose is not None else "missing")
                target_stream.write(json.dumps(record) + "\n")
                processed_valid_count += pose is not None
        if len(processed_file.read_text().splitlines()) != frame_count:
            raise ValueError(f"Processed frame count mismatch for {source}")
        if processed_valid_count - valid_count != row["interpolated_frames"]:
            raise ValueError(f"Interpolated frame count mismatch for {source}")
        exclusion_reasons = _policy_screen(row)
        metadata.append({
            "file_name": copied_video.name,
            "episode_id": episode_id,
            "recording_batch": ("initial_eight" if episode_id in INITIAL_RECORDING_IDS
                                else "additional_recordings"),
            "task": "Insert the USB stick into the USB port.",
            "usb_marker_size_mm": size_mm,
            "frames": frame_count,
            "valid_relative_poses": valid_count,
            "pose_file": f"../poses/{episode_id}.jsonl",
            "processed_pose_file": f"../poses_processed/{episode_id}.jsonl",
            "video_sha256": _sha256(copied_video),
            "audio_streams": 0,
            "video_processing": "video_stream_copy_audio_removed",
            "video_decoded_frames": video_frames,
            "trailing_untracked_video_frames": video_frames - frame_count,
            "tracking_status": row["status"],
            "replay_eligible": row["eligible"],
            "raw_valid_pose_fraction": row["valid_fraction"],
            "processed_valid_relative_poses": processed_valid_count,
            "interpolated_frames": row["interpolated_frames"],
            "tracking_decision": "replay_candidate" if row["eligible"] else "reject_insufficient_coverage",
            "tracking_rejection_reason": None if row["eligible"] else row.get("reason"),
            "geometric_replay_success": row["success"],
            "contact_free_replay_success": row["contact_free_success"],
            "wall_contact_steps": row.get("wall_contact_steps", 0),
            "policy_training_status": ("excluded" if exclusion_reasons else
                                       "candidate_for_manual_review"),
            "policy_exclusion_reasons": exclusion_reasons,
        })
    with (video_dir / "metadata.jsonl").open("w") as stream:
        for row in metadata:
            stream.write(json.dumps(row) + "\n")
    review_fields = ("episode_id", "usb_marker_size_mm", "frames", "raw_valid_pose_fraction",
                     "interpolated_frames", "replay_eligible", "geometric_replay_success",
                     "contact_free_replay_success", "wall_contact_steps",
                     "policy_training_status", "policy_exclusion_reasons")
    with (output_dir / "quality_review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_fields)
        writer.writeheader()
        for row in metadata:
            writer.writerow({field: (";".join(row[field]) if field == "policy_exclusion_reasons"
                                     else row[field]) for field in review_fields})
    card = f"""---
tags:
- robotics
- video
- aruco
---

# Human USB insertion: raw videos and tracked frames

This archive contains handheld iPhone videos of a person inserting a USB-A
stick into the **left** laptop port. The phone moves. ArUco marker 0 is on the
USB stick (12 or 20 mm outer black square); marker 1 is near the port (50 mm).
The iPhone calibration and printed marker sheets are in `calibration/`.
The archived `.mov` files contain **video only**: their video streams were
copied without re-encoding and all audio streams were removed. Original source
recordings in `video_data/` are unchanged. A container edit can expose up to
two extra trailing video frames; `video_decoded_frames` and
`trailing_untracked_video_frames` record this, and every tracked frame's
timestamp was checked against the archived video.

`videos/metadata.jsonl` follows Hugging Face VideoFolder conventions: each
`file_name` names a video beside the metadata file. The matching per-frame
pose JSONL is under `poses/`. `poses_processed/` contains a second trajectory
with at most four-frame gaps interpolated only when adjacent poses imply no
more than 0.25 m/s translation and 2 rad/s rotation. Each processed row has a
`pose_source` of `observed`, `interpolated`, or `missing`; original detections
are never overwritten. `frame_index` and `timestamp_s` align both files with
the source video. `quality` and marker reprojection errors must be checked.
Missing or rejected detections in the raw file have null target poses.

Tracking coverage after preprocessing: {sum(item['replay_eligible'] for item in metadata)}
of {len(metadata)} videos contain a qualifying insertion span. Clips with
`tracking_decision=reject_insufficient_coverage` remain in this archival
dataset but must not become robot-policy episodes. Short interpolation cannot
reconstruct the largely invisible 12 mm-marker trajectories.

`quality_review.csv` and `videos/metadata.jsonl` mark each clip with
`policy_training_status` and explicit `policy_exclusion_reasons`. An excluded
clip stays in this raw archive. `candidate_for_manual_review` means only that
the conservative automated checks passed; human review is still needed before
it enters a robot-policy dataset.

Frame convention: `T_A_B` maps coordinates in frame B into frame A; distances
are metres. OpenCV camera axes are x right, y down, z forward. The tracked
relative marker pose is `T_port_usb_marker = inverse(T_camera_port_marker) @
T_camera_usb_marker`. The fixed physical mounts in
`calibration/marker_mounts.json` give
`T_socket_usb_tip = T_socket_port_marker @ T_port_usb_marker @
T_usb_marker_usb_tip`. The port marker centre is `[+16, -45, 0]` mm in the
socket frame. The USB tip is `[0, +30, -3]` mm in the detected USB marker
frame. The socket origin is the left opening's centre, with +Z inward.
The target points were visually confirmed; orientation alignment was inferred
from the marker normals and a near-insertion frame.

These are **human videos and measured task poses**, not Panda demonstrations.
They contain no Panda joint states or robot actions and are not suitable as a
LeRobot robot-policy dataset by themselves. A separate MuJoCo replay dataset
will pair rendered policy images with actual simulated Panda states/actions.
The current evaluation status in `videos/metadata.jsonl` is descriptive. A
geometric replay success may still involve wall contact; inspect
`contact_free_replay_success` and `wall_contact_steps` before use as training
data. In this export, {sum(item['contact_free_replay_success'] for item in metadata)}
videos passed the contact-free replay check.

The footage may show a person and their surroundings. Review privacy and
choose a license before publishing this local export to a public Hub repo.
"""
    (output_dir / "README.md").write_text(card)
    result = {"videos": len(metadata), "frames": sum(item["frames"] for item in metadata),
              "new_videos_in_this_export": sum(item["episode_id"] not in previous_ids
                                               for item in metadata),
              "additional_recordings_since_initial_eight": sum(
                  item["recording_batch"] == "additional_recordings" for item in metadata),
              "valid_relative_poses": sum(item["valid_relative_poses"] for item in metadata),
              "processed_valid_relative_poses": sum(item["processed_valid_relative_poses"] for item in metadata),
              "replay_eligible": sum(item["replay_eligible"] for item in metadata),
              "contact_free_replays": sum(item["contact_free_replay_success"] for item in metadata),
              "policy_training_candidates": sum(item["policy_training_status"] ==
                                                "candidate_for_manual_review" for item in metadata),
              "videos_with_audio": sum(item["audio_streams"] > 0 for item in metadata),
              "output_dir": str(output_dir), "schema": "videofolder+pose_jsonl.v1"}
    (output_dir / "export_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
