"""Make a short queue for rechecking holds after the guided replay update."""

import csv
import json
from pathlib import Path


def main() -> None:
    root = Path("outputs/recalibration")
    decisions_path = Path("outputs/trial_review/decisions.csv")
    with decisions_path.open(newline="") as stream:
        decisions = {row["stem"]: row for row in csv.DictReader(stream)}
    primary = json.loads((root / "replay_results.json").read_text())
    fallback = {row["clip"]: row for row in json.loads((root / "fallback_results.json").read_text())}
    stability = {row["clip"]: row for row in json.loads((root / "marker_mount_stability.json").read_text())}
    contacts = {(row["clip"], row["tracking_mode"]): row
                for row in json.loads((root / "guided_contact_audit.json").read_text())}
    queue = []
    for row in primary:
        clip = row["clip"]
        decision = decisions.get(clip, {}).get("decision", "unreviewed")
        guided = bool(row.get("training_prefix_candidate"))
        recovered = bool(fallback.get(clip, {}).get("training_prefix_candidate"))
        priority = (0 if decision == "hold" and (guided or recovered) else
                    1 if decision == "unreviewed" and guided else
                    2 if decision == "unreviewed" and recovered else
                    3 if decision == "hold" else 4)
        queue.append({"priority": priority, "batch": row["batch"],
                      "clip": clip, "decision": decision,
                      "guided_primary_clean": guided, "marker0_fallback_clean": recovered,
                      "strict_entry": bool(row.get("strict_success")),
                      "approach_truncated": bool(row.get("approach_truncated")),
                      "early_late_marker_mount_shift_mm": stability.get(clip, {}).get(
                          "early_late_mount_translation_shift_mm"),
                      "guided_max_wall_force_n": contacts.get((clip, "primary"), {}).get(
                          "max_wall_force_n_before_success"),
                      "fallback_max_wall_force_n": contacts.get((clip, "marker0_fallback"), {}).get(
                          "max_wall_force_n_before_success"),
                      "source_track": row["source_track"]})
    queue.sort(key=lambda row: (row["priority"],
                                row["guided_max_wall_force_n"] if row["guided_max_wall_force_n"] is not None
                                else row["fallback_max_wall_force_n"] if row["fallback_max_wall_force_n"] is not None
                                else float("inf"), row["clip"]))
    with (root / "review_priority.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(queue[0]))
        writer.writeheader()
        writer.writerows(queue)
    print("Review queue:", sum(row["priority"] == 0 for row in queue),
          "prior holds now guided candidates,",
          sum(row["priority"] == 1 for row in queue), "unreviewed guided candidates")


if __name__ == "__main__":
    main()
