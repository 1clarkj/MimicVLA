"""Measure guided replay contacts up to first source-motion insertion."""

import csv
import json
from pathlib import Path


def main() -> None:
    root = Path("outputs/recalibration")
    primary = [row for row in json.loads((root / "replay_results.json").read_text())
               if row.get("training_prefix_candidate")]
    fallback = [row for row in json.loads((root / "fallback_results.json").read_text())
                if row.get("training_prefix_candidate")]
    audit = []
    for row, mode in ([(row, "primary") for row in primary] +
                      [(row, "marker0_fallback") for row in fallback]):
        folder = root / "replays" / row["batch"] / row["clip"]
        if mode == "marker0_fallback":
            folder /= "marker0_fallback"
        folder /= "training"
        summary = json.loads((folder / "summary.json").read_text())
        trajectory = json.loads((folder / "retargeted_replay.json").read_text())
        end = summary["initial_alignment_steps"] + summary["source_steps_to_first_success"]
        prefix = trajectory[:end + 1]
        audit.append({"batch": row["batch"], "clip": row["clip"], "tracking_mode": mode,
                      "max_wall_force_n_before_success": max(
                          frame["metrics"]["max_wall_normal_force_n"] for frame in prefix),
                      "max_funnel_penetration_mm_before_success": 1000 * max(
                          frame["metrics"].get("max_funnel_penetration_m", 0)
                          for frame in prefix),
                      "source_motion_frames": summary["source_steps_to_first_success"] + 1})
    (root / "guided_contact_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    with (root / "guided_contact_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    print("Candidates:", len(audit), ">100 N:",
          sum(row["max_wall_force_n_before_success"] > 100 for row in audit))


if __name__ == "__main__":
    main()
