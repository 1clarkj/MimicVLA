"""Build a local VideoFolder dataset of every human iPhone recording.

One row per clip in outputs/recalibration/replay_results.csv (134 clips).
Videos are stream-copied without audio; per-frame tracking is the immutable
source track plus the task-frame USB tip pose under the October recalibrated
session mounts (the exact poses replay_recalibrated.py replayed). Each row
gets a split: train / holdout (the 72-episode usb_15x7_no_funnel_full_v2
LeRobot dataset, keyed by -m mimicvla.dataset.splits), excluded_start_eval
(outputs/eval_sets/excluded_starts_v1) or unused (with a reason).

    python -m mimicvla.dataset.hf_export [--output DIR] [--overwrite]

Nothing is uploaded. A privacy contact sheet (start/middle/end frame per
clip) is written to --contact-sheet-dir.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import cv2
import imageio_ffmpeg
import numpy as np

from mimicvla.dataset.splits import HOLDOUT_CLIPS
from mimicvla.dataset.recordings import fill_short_gaps
from mimicvla.sim.scene import STRICT_SOCKET
from mimicvla.vision.target_frames import provisional_mounts


RECAL = Path("outputs/recalibration")
LEROBOT = Path("outputs/datasets/usb_15x7_no_funnel_full_v2")
EXCLUDED = Path("outputs/eval_sets/excluded_starts_v1")
DECISIONS = Path("outputs/trial_review/decisions.csv")
CALIBRATION = Path("outputs/iphone_calibration.json")
DUAL_MOUNTS = {"dual_original": Path("outputs/dual_marker_original_mount/usb_dual_mount.json"),
               "dual_minus3": Path("outputs/dual_marker_comparison/usb_dual_mount.json")}
MARKER_SETUP = {
    "single_12mm": ("12mm_marker", "single marker 0 (12 mm) on the USB stick"),
    "single_20mm": ("20mm_marker", "single marker 0 (20 mm) on the USB stick"),
    "dual_original": ("double_marker_original_mount",
                      "markers 0 and 10 on the USB stick, marker 0 at the original mount"),
    "dual_minus3": ("double_marker",
                    "markers 0 and 10 on the USB stick, second mount (USB tip prior offset -3 mm in x)"),
}
COMPANION = "<HF_USER>/mimicvla-panda-usb-15x7"
SHEET_CLIPS, THUMB_W = 20, 320


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_frame_times(path: Path) -> tuple[list[float], int]:
    """Presentation times of all video packets (no decode) and audio stream count."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        times = sorted(float(packet.pts * packet.time_base)
                       for packet in container.demux(stream) if packet.pts is not None)
        return [t - times[0] for t in times], len(container.streams.audio)


def remux(source: Path, target: Path, tracked_times: list[float]) -> int:
    """Stream-copy the video track without audio and check timestamp alignment."""
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                    "-nostdin", "-y", "-i", str(source), "-map", "0:v:0", "-c:v", "copy",
                    "-an", "-sn", "-dn", "-movflags", "+faststart", str(target)], check=True)
    times, audio = video_frame_times(target)
    if audio:
        raise ValueError(f"Audio left in {target}")
    if not len(tracked_times) <= len(times) <= len(tracked_times) + 2:
        raise ValueError(f"{target}: {len(times)} video frames vs {len(tracked_times)} tracked")
    if any(abs(a - b) > 0.015 for a, b in zip(times, tracked_times)):
        raise ValueError(f"{target}: video and tracking timestamps do not align")
    return len(times)


def longest_run(flags: list[bool]) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def as_bool(value: str):
    return {"True": True, "False": False, "": None}.get(value, value)


def as_number(value: str):
    if value in ("", None):
        return None
    number = float(value)
    return int(number) if number.is_integer() and "." not in value else number


def tracking_rows(raw: list[dict], port: np.ndarray, tip: np.ndarray, video: str) -> tuple[list[dict], dict]:
    filled, interpolated = fill_short_gaps(raw)
    rows, sources = [], []
    for record in filled:
        record = {key: value for key, value in record.items() if key != "rgb_file"}
        if record.get("interpolated"):
            source = "interpolated"
        elif record["quality"] == "ok":
            source = "observed"
        else:
            source = "missing"
        if source == "missing":
            record["T_port_usb_marker"] = None
            pose = None
        else:
            pose = (port @ np.asarray(record["T_port_usb_marker"], dtype=float) @ tip).tolist()
        record.update(pose_source=source, T_socket_usb_tip=pose, video_file=video)
        rows.append(record)
        sources.append(source)
    if sum(source == "interpolated" for source in sources) != interpolated:
        raise ValueError("Interpolation count mismatch")
    stats = {
        "observed_pose_frames": sources.count("observed"),
        "interpolated_pose_frames": interpolated,
        "missing_pose_frames": sources.count("missing"),
        "observed_pose_fraction": round(sources.count("observed") / len(sources), 4),
        "longest_missing_run_frames": longest_run([s == "missing" for s in sources]),
        "usb_pose_from_marker10_frames": sum(r.get("usb_pose_source") == "marker10" for r in raw),
        "task_orientation_prior_frames": sum(
            m.get("pose_selection") == "task_orientation_prior"
            for r in raw for key, m in (r.get("markers") or {}).items() if key in ("0", "10")),
    }
    return rows, stats


def extract_thumb(video: Path, time_s: float) -> np.ndarray:
    data = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                           "-nostdin", "-ss", f"{max(time_s, 0):.3f}", "-i", str(video),
                           "-frames:v", "1", "-vf", f"scale={THUMB_W}:-2", "-f", "image2pipe",
                           "-vcodec", "png", "-"], check=True, capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def contact_sheets(rows: list[dict], output: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("contact_sheet_*.jpg"):
        old.unlink()
    jobs = []
    for row in rows:
        video = output / "videos" / row["file_name"]
        last = row["video_duration_s"]
        jobs += [(video, t) for t in (0.0, last / 2, max(last - 0.1, 0.0))]
    with ThreadPoolExecutor(8) as pool:
        thumbs = list(pool.map(lambda job: extract_thumb(*job), jobs))
    height = max(t.shape[0] for t in thumbs)
    width = max(t.shape[1] for t in thumbs)
    label_h = 24
    tiles = []
    for index, row in enumerate(rows):
        tile = np.full((height + label_h, 3 * width, 3), 255, np.uint8)
        for k in range(3):
            thumb = thumbs[3 * index + k]
            tile[label_h:label_h + thumb.shape[0], k * width:k * width + thumb.shape[1]] = thumb
        cv2.putText(tile, f"{row['clip_id']}  ({row['batch']}, {row['split']})", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    paths = []
    for sheet, start in enumerate(range(0, len(tiles), SHEET_CLIPS)):
        chunk = tiles[start:start + SHEET_CLIPS]
        chunk += [np.full_like(tiles[0], 255)] * (-len(chunk) % 2)
        grid = np.vstack([np.hstack([chunk[i], np.full((chunk[i].shape[0], 8, 3), 128, np.uint8),
                                     chunk[i + 1]]) for i in range(0, len(chunk), 2)])
        path = out_dir / f"contact_sheet_{sheet + 1:02d}.jpg"
        cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(path)
    return paths


def card(rows: list[dict], counts: dict, wide: dict, size_bytes: int) -> str:
    license_name = "mit"
    strict = STRICT_SOCKET
    by_batch = {}
    for row in rows:
        by_batch[row["batch"]] = by_batch.get(row["batch"], 0) + 1
    batch_lines = "\n".join(
        f"| `{batch}` | `video_data/{MARKER_SETUP[batch][0]}/` | {MARKER_SETUP[batch][1]} | {n} |"
        for batch, n in by_batch.items())
    reviewed = sum(row["human_review_decision"] not in (None, "unreviewed") for row in rows)
    return f"""---
license: {license_name}
task_categories:
- robotics
tags:
- robotics
- video
- aruco
- human-demonstration
- usb-insertion
- pose-estimation
pretty_name: MimicVLA human USB insertion (raw iPhone videos and tracking)
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: videos/*
---

# MimicVLA human USB insertion: raw videos and tracking

{len(rows)} handheld iPhone videos of one person inserting a USB-A stick into
the left USB port on the front panel of a desktop computer. The phone is held in the other hand and moves
during each clip. ArUco markers (`DICT_4X4_50`) are attached to the plug and
next to the port; their per-frame 6-DoF poses were estimated from the video and
converted into a USB-tip pose in a port-fixed frame. These poses were
retargeted to a simulated Franka Panda in MuJoCo to produce the companion
LeRobot dataset `{COMPANION}`.

This dataset contains human video and measured poses only. It has no robot
joint states or robot actions.

## Contents

| Path | Content |
| --- | --- |
| `videos/<clip_id>.mov` | Original iPhone video stream (HEVC, 3840x2160, about 24 fps), stream-copied without re-encoding; audio removed |
| `videos/metadata.jsonl` | One row per clip (VideoFolder metadata) |
| `tracking/<clip_id>.jsonl` | One row per tracked video frame |
| `calibration/iphone_calibration.json` | Camera intrinsics and distortion (OpenCV model) |
| `calibration/marker_mounts.json` | Port mount, USB-tip mount per calibration session, dual-marker mounts |
| `calibration/candidate_mounts.json` | Full recalibration report (per-session fits and residuals) |
| `calibration/usb_dual_mount_*.json` | Fixed transform between USB markers 0 and 10 for the dual-marker batches |
| `calibration/*.pdf` | Printed ChArUco calibration board and ArUco marker sheet (A4) |

Recording batches:

| batch | source folder | USB markers | clips |
| --- | --- | --- | --- |
{batch_lines}

Marker 1 (50 mm) is on the computer's front panel next to the port in every batch.

## Frame conventions and units

- `T_A_B` is a 4x4 homogeneous transform that maps coordinates in frame B into
  frame A. Translations are in metres, rotations are rotation matrices.
- Camera frame: OpenCV convention, x right, y down, z forward (optical axis).
- `markers[id].T_camera_marker`: pose of each detected marker in the camera
  frame. Marker frame: origin at the marker centre, +z out of the printed face.
- `T_port_usb_marker = inverse(T_camera_port_marker) @ T_camera_usb_marker`,
  where the port marker is marker 1 and the USB marker frame is marker 0.
  When only marker 10 is visible on a dual-marker clip, marker 10's pose is
  mapped into marker 0's frame with the fixed dual mount (`usb_pose_source`).
- `T_socket_usb_tip = T_socket_port_marker @ T_port_usb_marker @
  T_usb_marker_usb_tip`. The socket frame has its origin at the centre of the
  port opening and +z pointing into the computer. The port marker centre is at
  [+16, -45, 0] mm in the socket frame. The USB tip mount is
  [x, +30, z] mm in marker 0's frame; x and z were refit per recording
  session from clips in which the plug was seen to enter the port
  (`calibration_epoch`, `T_usb_marker_usb_tip` in the metadata). This is a
  task calibration with about 1-2 mm residual, not a measured ground truth.
- `timestamp_s` is the video presentation time; `frame_index` counts decoded
  video frames from 0. Each tracked timestamp was checked against the exported
  video (within 15 ms). Up to two trailing video frames can be untracked.

## Tracking files

Each row of `tracking/<clip_id>.jsonl` has `frame_index`, `timestamp_s`,
`visible_ids`, `markers` (pose, reprojection error, pose ambiguity gap and how
the planar pose ambiguity was resolved), `quality`, `usb_pose_source`,
`T_port_usb_marker`, `T_socket_usb_tip` and `pose_source`:

- `observed`: pose measured in this frame (`quality == "ok"`).
- `interpolated`: gap of at most 4 frames filled by linear/slerp
  interpolation, only when the neighbouring poses imply at most 0.25 m/s and
  2 rad/s. These rows have `quality == "ok"`, `interpolated == true` and the
  original failure label in `quality_original`.
- `missing`: no pose; `T_port_usb_marker` and `T_socket_usb_tip` are null.

These are the poses used for the MuJoCo replays. The replays additionally
low-pass filter and resample the selected segment.

## Metadata fields (videos/metadata.jsonl)

- `clip_id`, `batch`, `marker_setup`, `usb_marker_size_mm`, `source_file`.
- `video_frames`, `tracked_frames`, `trailing_untracked_video_frames`,
  `video_duration_s`, `video_sha256`.
- Tracking statistics: `observed_pose_frames`, `interpolated_pose_frames`,
  `missing_pose_frames`, `observed_pose_fraction`,
  `longest_missing_run_frames`, `usb_pose_from_marker10_frames`,
  `task_orientation_prior_frames` (marker detections where a task prior chose
  between the two planar pose solutions).
- `tracking_status`: `replayable` if the track contains a continuous
  insertion segment (at least 20 frames with 10 mm progress, reaching 6 mm
  depth), else `insufficient_continuous_tracking` with `tracking_reason`.
  `replay_start_frame` / `replay_end_frame` delimit that segment.
- `human_review_*`: manual review of each clip in a browser tool that showed
  the source video and the replay. `decision` is accept / reject / hold /
  unreviewed; `human_success` is whether the person in the video inserted
  the USB stick; `tracking` and `replay` are optional quality notes
  (mostly `unknown`); `notes` is free text (mostly whether the clip passed an
  earlier, pre-recalibration replay). Missing for clips never reviewed.
- `review_replay_*`: replay in the {wide['inner_half_x_m'] * 2000:.0f} x
  {wide['inner_half_y_m'] * 2000:.0f} mm no-funnel port. This
  replay decides whether the robot reached the aligned start pose.
- `replay_15x7_*`: replay of the complete source motion into a
  15 x 7 mm no-funnel port (see below). Null when the clip was not replayed in
  that geometry (it failed review or the 7 mm entry screen).
- `split`, `split_reason`, `lerobot_episode_index`,
  `excluded_start_eval_episode_index`.

## Splits

All clips are in one Hugging Face split (`train`); use the `split` column:

| split | count | definition |
| --- | --- | --- |
| `train` | {counts.get('train', 0)} | episodes of `{COMPANION}` that are not held out |
| `holdout` | {counts.get('holdout', 0)} | episodes of `{COMPANION}` held out for policy evaluation (fixed list, stratified by batch, numpy seed 2026) |
| `excluded_start_eval` | {counts.get('excluded_start_eval', 0)} | not in the LeRobot dataset; the Panda replay reached the aligned start pose, and that start state was used only to evaluate policies on unseen starts |
| `unused` | {counts.get('unused', 0)} | see `split_reason` |

`lerobot_episode_index` is the episode index of the clip in `{COMPANION}`
(train and holdout only).

## Human success vs replay success

Human success means the person inserted the physical USB stick into the
physical port. Every reviewed clip shows a successful human insertion.

Replay success is a property of the simulation, not of the human motion. The
tracked `T_socket_usb_tip` trajectory is replayed by an operational-space
controller on a simulated Franka Panda in MuJoCo, holding a USB plug, into a
rectangular port without an entry funnel. `replay_15x7_success` is true when,
in a 15 x 7 mm port, the tip centre passes 7 mm depth during the source
motion, the whole replay has no flagged deep interpenetration or joint-limit
violation, and the plug finishes at least 12 mm deep. The physical plug is
about 12 x 4.5 mm, so the replay port has 1.5 mm clearance per side in x and
1.25 mm in y, wider than a real port (strict model: {strict.inner_half_x_m * 2000:.1f} x
{strict.inner_half_y_m * 2000:.1f} mm). A human success can fail in replay because of
tracking error, calibration residual, or the robot's reach and contact
behaviour. Wall contact during insertion is allowed.

Clips enter the LeRobot dataset only if they were accepted in human review
and are replay successes.

## Collection

Recorded in October 2026 with an iPhone (4K, about 24 fps), handheld. One
person, one computer, one USB stick. {reviewed} of {len(rows)} clips were
manually reviewed. The videos show a hand, the computer and its surroundings;
check the footage before reuse.

Total size: {size_bytes / 1e9:.2f} GB.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("outputs/hf_upload/mimicvla-human-usb-raw"))
    parser.add_argument("--contact-sheet-dir", type=Path, default=Path("outputs/hf_upload/contact_sheets"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{args.output} exists; pass --overwrite to rebuild it")
        shutil.rmtree(args.output)
    for folder in ("videos", "tracking", "calibration"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)

    with (RECAL / "replay_results.csv").open(newline="") as stream:
        replay_rows = list(csv.DictReader(stream))
    with DECISIONS.open(newline="") as stream:
        decisions = {row["stem"]: row for row in csv.DictReader(stream)}
    lerobot = {e["clip"]: e for e in
               json.loads((LEROBOT / "capture_manifest.json").read_text())["episodes_manifest"]}
    excluded = {e["clip"]: e for e in
                json.loads((EXCLUDED / "capture_manifest.json").read_text())["episodes_manifest"]}
    audit = {e["clip"]: e for e in
             json.loads((RECAL / "compact_15x7_full_replay_audit.json").read_text())["episodes"]}
    sweep = json.loads((RECAL / "no_funnel_7mm_port_sweep.json").read_text())
    entry = {e["clip"]: e for size in sweep["sizes"]
             if (size["width_mm"], size["height_mm"]) == (15.0, 7.0) for e in size["episodes"]}
    mounts = json.loads((RECAL / "candidate_mounts.json").read_text())
    port, _ = provisional_mounts()
    if len(replay_rows) != 134 or len({r["clip"] for r in replay_rows}) != 134:
        raise ValueError("Expected 134 unique clips in replay_results.csv")

    def epoch_for(row):
        for epoch in mounts["batches"][row["batch"]]["epochs"]:
            if (epoch["first_clip"], epoch["last_clip"]) == (row["epoch_first"], row["epoch_last"]):
                return epoch
        raise ValueError(f"No calibration epoch for {row['clip']}")

    def build(row: dict) -> dict:
        clip, batch = row["clip"], row["batch"]
        track = Path(row["source_track"])
        source = Path(json.loads((track.parent / "metadata.json").read_text())["source"])
        if source.parent.name != MARKER_SETUP[batch][0]:
            raise ValueError(f"{clip}: source {source} does not match batch {batch}")
        raw = [json.loads(line) for line in track.read_text().splitlines()]
        epoch = epoch_for(row)
        tip = np.asarray(epoch["T_marker0_usb_tip"], dtype=float)
        video = args.output / "videos" / f"{clip}.mov"
        frames = remux(source, video, [r["timestamp_s"] for r in raw])
        tracked, stats = tracking_rows(raw, port, tip, f"videos/{video.name}")
        with (args.output / "tracking" / f"{clip}.jsonl").open("w") as stream:
            for record in tracked:
                stream.write(json.dumps(record) + "\n")
        replay_dir = RECAL / "replays" / batch / clip / "wide_no_funnel"
        wide = (json.loads((replay_dir / "summary.json").read_text())
                if (replay_dir / "summary.json").exists() else None)
        review = decisions.get(clip)
        key = (batch, clip)
        if clip in lerobot:
            split = "holdout" if key in HOLDOUT_CLIPS else "train"
            reason = "episode of the 15x7 LeRobot dataset"
        elif clip in excluded:
            split, reason = "excluded_start_eval", (
                "not in the LeRobot dataset; replay start state used for policy evaluation")
        elif row["status"] != "replayable":
            split, reason = "unused", "insufficient_continuous_tracking"
        elif wide is not None and not wide["alignment_reached"]:
            split, reason = "unused", "replay_did_not_reach_aligned_start"
        else:
            split, reason = "unused", "unclassified"
        audited = audit.get(clip)
        success_15x7 = None if audited is None else bool(
            entry[clip]["valid"] and audited["final_depth_mm"] >= 12
            and audited["first_bad_collision_step"] is None
            and audited["first_joint_limit_step"] is None)
        times = [r["timestamp_s"] for r in raw]
        return {
            "file_name": video.name,
            "clip_id": clip,
            "batch": batch,
            "marker_setup": MARKER_SETUP[batch][1],
            "usb_marker_size_mm": 12 if batch == "single_12mm" else 20,
            "source_file": str(source),
            "task": "Insert the USB stick into the USB port.",
            "split": split,
            "split_reason": reason,
            "lerobot_episode_index": lerobot[clip]["episode_index"] if clip in lerobot else None,
            "excluded_start_eval_episode_index": (excluded[clip]["episode_index"]
                                                  if clip in excluded else None),
            "tracking_file": f"tracking/{clip}.jsonl",
            "video_frames": frames,
            "tracked_frames": len(raw),
            "trailing_untracked_video_frames": frames - len(raw),
            "video_duration_s": round(times[-1] + (times[-1] - times[-2]), 4),
            "video_sha256": sha256(video),
            **stats,
            "calibration_epoch": [epoch["first_clip"], epoch["last_clip"]],
            "calibration_epoch_borrowed": as_bool(row["borrowed_epoch"]),
            "T_usb_marker_usb_tip": tip.tolist(),
            "tracking_status": row["status"],
            "tracking_reason": row["reason"] or None,
            "replay_start_frame": as_number(row.get("start_frame", "")),
            "replay_end_frame": as_number(row.get("end_frame", "")),
            "human_review_decision": review["decision"] if review else "unreviewed",
            "human_review_human_success": review["human_success"] if review else None,
            "human_review_tracking": review["tracking"] if review else None,
            "human_review_replay": review["replay"] if review else None,
            "human_review_notes": (review["notes"] or None) if review else None,
            "review_replay_insertion_reached": None if wide is None else wide["insertion_reached"],
            "review_replay_alignment_reached": None if wide is None else wide["alignment_reached"],
            "review_replay_max_depth_mm": (None if wide is None else
                                           round(wide["max_axial_depth_m"] * 1000, 3)),
            "review_replay_bad_collision_steps": None if wide is None else wide["bad_collision_steps"],
            "review_replay_joint_limits_ok": None if wide is None else wide["all_joint_limits_ok"],
            "replay_15x7_entry_screen_passed": (entry[clip]["valid"] if clip in entry else None),
            "replay_15x7_success": success_15x7,
            "replay_15x7_final_depth_mm": (None if audited is None else
                                           round(audited["final_depth_mm"], 3)),
            "replay_15x7_peak_wall_force_n": (None if audited is None else
                                              round(audited["peak_wall_force_n"], 1)),
            "excluded_start_source_replay_reached_15x7_insertion": (
                excluded[clip]["source_replay_reached_15x7_insertion"] if clip in excluded else None),
        }

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(build, replay_rows))
    rows.sort(key=lambda r: (list(MARKER_SETUP).index(r["batch"]), r["clip_id"]))
    for clip, episode in lerobot.items():
        row = next(r for r in rows if r["clip_id"] == clip)
        if row["replay_15x7_success"] is not True or row["human_review_decision"] != "accept":
            raise ValueError(f"LeRobot episode {clip} is not an accepted 15x7 replay success")
    with (args.output / "videos" / "metadata.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")

    calibration = args.output / "calibration"
    shutil.copy2(CALIBRATION, calibration / "iphone_calibration.json")
    shutil.copy2(RECAL / "candidate_mounts.json", calibration / "candidate_mounts.json")
    for pdf in sorted(Path("calibration").glob("*.pdf")):
        shutil.copy2(pdf, calibration / pdf.name)
    for batch, path in DUAL_MOUNTS.items():
        shutil.copy2(path, calibration / f"usb_dual_mount_{batch}.json")
    (calibration / "marker_mounts.json").write_text(json.dumps({
        "convention": "T_A_B maps coordinates in B into A; translations in metres",
        "markers": {"0": "USB stick (12 or 20 mm)", "10": "USB stick, dual-marker batches",
                    "1": "computer front panel next to the port (50 mm)"},
        "T_socket_port_marker": port.tolist(),
        "socket_frame": "origin at the centre of the left USB port opening, +z into the computer",
        "marker_frame": "origin at the marker centre, +z out of the printed face",
        "T_usb_marker_usb_tip_per_session": {
            batch: [{"first_clip": e["first_clip"], "last_clip": e["last_clip"],
                     "T_marker0_usb_tip": e["T_marker0_usb_tip"]}
                    for e in info["epochs"]] for batch, info in mounts["batches"].items()},
        "dual_mount_files": {batch: f"usb_dual_mount_{batch}.json" for batch in DUAL_MOUNTS},
        "note": mounts["caveat"],
    }, indent=2) + "\n")

    counts = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    wide_geometry = json.loads(next((RECAL / "replays").glob("*/*/wide_no_funnel/summary.json"))
                               .read_text())["socket_geometry"]
    size = sum(f.stat().st_size for f in args.output.rglob("*") if f.is_file())
    (args.output / "README.md").write_text(card(rows, counts, wide_geometry, size))
    sheets = contact_sheets(rows, args.output, args.contact_sheet_dir)
    print(json.dumps({"output": str(args.output), "rows": len(rows), "splits": counts,
                      "bytes": size, "contact_sheets": [str(p) for p in sheets]}, indent=2))


if __name__ == "__main__":
    main()
