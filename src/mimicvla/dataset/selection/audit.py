"""Audit the entire accepted 15 x 7 mm replay after the 7 mm selection point."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mimicvla.dataset.selection.export import DEPTH_M, GEOMETRY
from mimicvla.sim.env import PandaUSBEnv, SimConfig, insertion_success_from_metrics


def main() -> None:
    source = json.loads(Path("outputs/datasets/usb_15x7_no_funnel_reviewed_7mm_v1/"
                             "capture_manifest.json").read_text())
    rows = []
    for episode in source["episodes_manifest"]:
        path = (Path("outputs/recalibration/replays") / episode["batch"] /
                episode["clip"] / "wide_no_funnel")
        replay = json.loads((path / "retargeted_replay.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        first = summary["initial_alignment_steps"]
        env = PandaUSBEnv(SimConfig(max_steps=1000, geometry=GEOMETRY,
                                    insertion_depth_m=DEPTH_M))
        first_7 = None
        first_12 = None
        first_bad = None
        first_joint = None
        max_depth_m = -float("inf")
        max_depth_step = None
        peak_force_n = 0.0
        try:
            for index, entry in enumerate(replay):
                _, _, info = env.step(np.asarray(entry["action"], dtype=float))
                if index < first:
                    continue
                if info["insertion_success"] and first_7 is None:
                    first_7 = index
                if (insertion_success_from_metrics(info, required_depth_m=.012,
                                                   geometry=GEOMETRY) and first_12 is None):
                    first_12 = index
                if info["bad_collision"] and first_bad is None:
                    first_bad = index
                joints = env.observation(render=False)["observation.state"][:7]
                limits = env.model.jnt_range[:7]
                if (first_joint is None and
                        (np.any(joints < limits[:, 0] - 1e-4) or
                         np.any(joints > limits[:, 1] + 1e-4))):
                    first_joint = index
                if info["axial_error_m"] > max_depth_m:
                    max_depth_m = info["axial_error_m"]
                    max_depth_step = index
                peak_force_n = max(peak_force_n, info["max_wall_normal_force_n"])
            rows.append({"batch": episode["batch"], "clip": episode["clip"],
                         "first_source_step": first, "last_source_step": len(replay) - 1,
                         "first_7mm_step": first_7, "first_12mm_step": first_12,
                         "first_bad_collision_step": first_bad,
                         "first_joint_limit_step": first_joint,
                         "max_depth_mm": max_depth_m * 1000,
                         "max_depth_step": max_depth_step,
                         "final_depth_mm": info["axial_error_m"] * 1000,
                         "final_7mm_success": info["insertion_success"],
                         "final_12mm_success": insertion_success_from_metrics(
                             info, required_depth_m=.012, geometry=GEOMETRY),
                         "peak_wall_force_n": peak_force_n})
        finally:
            env.close()
    result = {"geometry": vars(GEOMETRY), "selection_depth_mm": 7,
              "episodes": rows}
    output = Path("outputs/recalibration/compact_15x7_full_replay_audit.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"episodes": len(rows),
                      "reach_12mm": sum(row["first_12mm_step"] is not None for row in rows),
                      "bad_collision_any": sum(row["first_bad_collision_step"] is not None
                                               for row in rows),
                      "joint_limit_any": sum(row["first_joint_limit_step"] is not None
                                             for row in rows),
                      "end_past_7mm": sum(row["final_depth_mm"] >= 7 for row in rows),
                      "end_7mm_success": sum(row["final_7mm_success"] for row in rows),
                      "end_12mm_success": sum(row["final_12mm_success"] for row in rows),
                      "max_depth_past_12mm": sum(row["max_depth_mm"] >= 12 for row in rows)},
                     indent=2))


if __name__ == "__main__":
    main()
