"""Try marker 0 alone when the fused dual-marker replay misses the guided port."""

import json
from pathlib import Path

import numpy as np

from mimicvla.evaluation.recordings import fill_short_gaps, select_insertion_run
from mimicvla.retarget import replay_recording
from mimicvla.sim.env import SimConfig
from mimicvla.sim.scene import TRAINING_SOCKET


def main() -> None:
    root = Path("outputs/recalibration")
    primary = [row for row in json.loads((root / "replay_results.json").read_text())
               if row["batch"] == "dual_original" and not row.get("training_prefix_candidate")]
    epochs = json.loads((root / "candidate_mounts.json").read_text())["batches"]["dual_original"]["epochs"]
    results = []
    for row in primary:
        clip = row["clip"]
        epoch = next(epoch for epoch in epochs
                     if epoch["first_clip"] <= clip <= epoch["last_clip"])
        tip = np.asarray(epoch["T_marker0_usb_tip"])
        source = Path("outputs/dual_marker_original_mount/marker0_only") / clip / "tracking/trajectory.jsonl"
        records = [json.loads(line) for line in source.read_text().splitlines()]
        filled, interpolated = fill_short_gaps(records)
        segment, detail = select_insertion_run(filled, marker_tip=tip)
        result = {"batch": "dual_original", "clip": clip, "tracking_mode": "marker0_only",
                  "source_track": str(source), "interpolated_frames": interpolated,
                  "epoch_first": epoch["first_clip"], "epoch_last": epoch["last_clip"],
                  "needs_extra_visual_review": True, **detail}
        if segment is None:
            result["status"] = "insufficient_continuous_tracking"
        else:
            result.update(status="replayable", start_frame=segment[0]["frame_index"],
                          end_frame=segment[-1]["frame_index"])
            folder = root / "replays" / "dual_original" / clip / "marker0_fallback"
            folder.mkdir(parents=True, exist_ok=True)
            filled_path = folder / "trajectory_filled.jsonl"
            filled_path.write_text("".join(json.dumps(record) + "\n" for record in filled))
            summary = replay_recording(
                filled_path, folder / "training",
                start_frame=result["start_frame"], end_frame=result["end_frame"],
                complete_segment=True, marker_tip=tip,
                sim_config=SimConfig(max_steps=1000, geometry=TRAINING_SOCKET),
                write_video=False)
            result.update(training_success=summary["source_motion_insertion_reached"],
                          training_prefix_candidate=bool(
                              summary["source_motion_insertion_reached"]
                              and summary["source_prefix_joint_limits_ok"]
                              and summary["source_prefix_bad_collision_steps"] == 0),
                          training_bad_collision_steps=summary["bad_collision_steps"],
                          training_max_depth_mm=summary["max_axial_depth_m"] * 1000)
        results.append(result)
        print(clip, result.get("training_prefix_candidate", False), flush=True)
    (root / "fallback_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Recovered", sum(row.get("training_prefix_candidate", False) for row in results),
          "of", len(results), "dual-marker misses")


if __name__ == "__main__":
    main()
