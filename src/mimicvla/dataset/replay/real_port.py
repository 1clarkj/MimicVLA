"""Replay the recalibrated tracks on the real 13.4 x 5.9 mm port and record the scores."""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path

import numpy as np

from mimicvla.dataset.tracking.recordings import fill_short_gaps, select_insertion_run
from mimicvla.retarget import replay_recording
from mimicvla.sim.env import SimConfig
from mimicvla.sim.scene import STRICT_SOCKET

from mimicvla.dataset.tracking.recalibrate import BATCHES


def choose_epoch(clip: str, epochs: list[dict]) -> tuple[dict, bool]:
    for epoch in epochs:
        if epoch["first_clip"] <= clip <= epoch["last_clip"]:
            return epoch, "borrowed_from" in epoch
    # Tracks without enough valid interior frames still need a session mount.
    return min(epochs, key=lambda item: min(abs(int(clip.split("_")[-1]) -
                                             int(item["first_clip"].split("_")[-1])),
                                         abs(int(clip.split("_")[-1]) -
                                             int(item["last_clip"].split("_")[-1])))), True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=[*BATCHES, "all"], default="all")
    args = parser.parse_args()
    root = Path("outputs/recalibration")
    calibration = json.loads((root / "candidate_mounts.json").read_text())
    decisions = {}
    decisions_path = Path("outputs/trial_review/decisions.csv")
    if decisions_path.exists():
        with decisions_path.open(newline="") as stream:
            decisions = {row["stem"]: row for row in csv.DictReader(stream)}
    results = []
    batches = BATCHES if args.batch == "all" else {args.batch: BATCHES[args.batch]}
    # The real 13.4 x 5.9 mm port; the widened port is replayed by
    # mimicvla.dataset.replay.wide_port and mimicvla.dataset.replay_additional.
    geometries = {"strict": STRICT_SOCKET}
    for batch_name, (pattern, _) in batches.items():
        epochs = calibration["batches"][batch_name]["epochs"]
        if not epochs:
            continue
        for path in sorted(Path().glob(pattern)):
            clip = path.parent.parent.name
            epoch, borrowed = choose_epoch(clip, epochs)
            tip = np.asarray(epoch["T_marker0_usb_tip"], dtype=float)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            filled, interpolated = fill_short_gaps(records)
            segment, detail = select_insertion_run(filled, marker_tip=tip)
            row = {"batch": batch_name, "clip": clip, "source_track": str(path),
                   "epoch_first": epoch["first_clip"], "epoch_last": epoch["last_clip"],
                   "borrowed_epoch": borrowed, "interpolated_frames": interpolated,
                   "human_review_decision": decisions.get(clip, {}).get("decision", "unreviewed"),
                   "human_review_tracking": decisions.get(clip, {}).get("tracking", "unreviewed"),
                   **detail}
            out = root / "replays" / batch_name / clip
            out.mkdir(parents=True, exist_ok=True)
            if segment is None:
                row["status"] = "insufficient_continuous_tracking"
            else:
                row.update(status="replayable", start_frame=segment[0]["frame_index"],
                           end_frame=segment[-1]["frame_index"],
                           segment_frames=len(segment))
                filled_path = out / "trajectory_filled.jsonl"
                with filled_path.open("w") as stream:
                    for record in filled:
                        stream.write(json.dumps(record) + "\n")
                for mode, geometry in geometries.items():
                    try:
                        summary = replay_recording(
                            filled_path, out / mode,
                            start_frame=segment[0]["frame_index"],
                            end_frame=segment[-1]["frame_index"],
                            complete_segment=True, marker_tip=tip,
                            sim_config=SimConfig(max_steps=1000, geometry=geometry),
                            write_video=False)
                        row[f"{mode}_success"] = summary["insertion_reached"]
                        row[f"{mode}_max_depth_mm"] = summary["max_axial_depth_m"] * 1000
                        row[f"{mode}_bad_collision_steps"] = summary["bad_collision_steps"]
                        row[f"{mode}_wall_contact_steps"] = summary["wall_contact_steps"]
                        row[f"{mode}_joint_limits_ok"] = summary["all_joint_limits_ok"]
                        row[f"{mode}_completed"] = summary["completed_segment"]
                        row[f"{mode}_source_motion_success"] = summary["source_motion_insertion_reached"]
                        row[f"{mode}_initial_alignment_steps"] = summary["initial_alignment_steps"]
                        row[f"{mode}_source_prefix_bad_collision_steps"] = summary["source_prefix_bad_collision_steps"]
                        row[f"{mode}_prefix_candidate"] = bool(
                            summary["source_motion_insertion_reached"]
                            and summary["source_prefix_joint_limits_ok"]
                            and summary["source_prefix_bad_collision_steps"] == 0)
                        row[f"{mode}_training_candidate"] = bool(
                            summary["insertion_reached"] and summary["completed_segment"]
                            and summary["all_joint_limits_ok"]
                            and summary["bad_collision_steps"] == 0)
                    except Exception as error:
                        row[f"{mode}_error"] = f"{type(error).__name__}: {error}"
                        (out / f"{mode}_error.txt").write_text(traceback.format_exc())
            results.append(row)
            (root / "replay_results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(batch_name, clip, row["status"],
                  *(f"{name}={row.get(f'{name}_success', 'error')}" for name in geometries),
                  flush=True)
    fields = sorted({key for row in results for key in row})
    with (root / "replay_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    summary = {"geometries": {name: vars(geometry) for name, geometry in geometries.items()},
               "batches": {}}
    for batch_name in batches:
        rows = [r for r in results if r["batch"] == batch_name]
        summary["batches"][batch_name] = {"clips": len(rows),
                                           "replayable": sum(r["status"] == "replayable" for r in rows),
                                           "approach_truncated": sum(r.get("approach_truncated", False) for r in rows),
                                           "reviewed_accept": sum(r["human_review_decision"] == "accept" for r in rows)}
        for mode in geometries:
            summary["batches"][batch_name][mode] = {
                "successes": sum(r.get(f"{mode}_success", False) for r in rows),
                "training_candidates": sum(r.get(f"{mode}_training_candidate", False) for r in rows),
                "clean_source_prefixes": sum(r.get(f"{mode}_prefix_candidate", False) for r in rows),
                "reviewed_accept_candidates": sum(r.get(f"{mode}_training_candidate", False)
                                                  and r["human_review_decision"] == "accept"
                                                  for r in rows),
                "errors": sum(f"{mode}_error" in r for r in rows)}
    (root / "replay_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
