"""Evaluate single-marker and older dual-marker tracks on the widened port."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mimicvla.retarget import replay_recording
from mimicvla.sim.env import SimConfig
from mimicvla.sim.scene import SocketGeometry
from mimicvla.dataset.replay import choose_epoch


GEOMETRY = SocketGeometry(inner_half_x_m=.009, inner_half_y_m=.005,
                          mouth_half_x_m=.009, mouth_half_y_m=.005)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=("single_20mm", "dual_minus3", "all"),
                        default="all")
    args = parser.parse_args()
    root = Path("outputs/recalibration")
    rows = json.loads((root / "replay_results.json").read_text())
    calibration = json.loads((root / "candidate_mounts.json").read_text())
    batches = ("single_20mm", "dual_minus3") if args.batch == "all" else (args.batch,)
    results = []
    output = root / "additional_wide_no_funnel_results.json"
    for batch in batches:
        epochs = calibration["batches"][batch]["epochs"]
        for row in (item for item in rows if item["batch"] == batch):
            result = {"batch": batch, "clip": row["clip"], "status": row["status"],
                      "source_track": row["source_track"],
                      "approach_truncated": row.get("approach_truncated", False)}
            if row["status"] == "replayable":
                folder = root / "replays" / batch / row["clip"]
                epoch, _ = choose_epoch(row["clip"], epochs)
                summary = replay_recording(
                    folder / "trajectory_filled.jsonl", folder / "wide_no_funnel",
                    start_frame=row["start_frame"], end_frame=row["end_frame"],
                    complete_segment=True,
                    marker_tip=np.asarray(epoch["T_marker0_usb_tip"], dtype=float),
                    sim_config=SimConfig(max_steps=1000, geometry=GEOMETRY),
                    write_video=False)
                clean = (summary["source_motion_insertion_reached"] and
                         summary["source_prefix_joint_limits_ok"] and
                         summary["source_prefix_bad_collision_steps"] == 0)
                result.update(entry=summary["insertion_reached"], clean_prefix=clean,
                              max_depth_mm=summary["max_axial_depth_m"] * 1000)
                if clean:
                    steps = json.loads((folder / "wide_no_funnel" /
                                        "retargeted_replay.json").read_text())
                    last = (summary["initial_alignment_steps"] +
                            summary["source_steps_to_first_success"])
                    result["peak_contact_n_to_first_entry"] = max(
                        step["metrics"]["max_wall_normal_force_n"]
                        for step in steps[:last + 1])
            results.append(result)
            output.write_text(json.dumps(results, indent=2) + "\n")
            print(batch, row["clip"], result.get("clean_prefix", False), flush=True)
    print(json.dumps({batch: {"clips": sum(row["batch"] == batch for row in results),
                              "clean_prefixes": sum(row["batch"] == batch and
                                                    row.get("clean_prefix", False)
                                                    for row in results),
                              "under_100_n": sum(row["batch"] == batch and
                                                 row.get("clean_prefix", False) and
                                                 row["peak_contact_n_to_first_entry"] <= 100
                                                 for row in results)} for batch in batches}, indent=2))


if __name__ == "__main__":
    main()
