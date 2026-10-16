"""Build a review queue from the saved widened-port, no-funnel replays."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("outputs/recalibration")
FORCE_SCREEN_N = 100.0  # Dataset triage heuristic, not a hardware safety limit.
MOUNT_SHIFT_SCREEN_MM = 4.0


def main() -> None:
    replay_rows = [row for row in json.loads((ROOT / "replay_results.json").read_text())
                   if row["batch"] == "dual_original"]
    stability = {row["clip"]: row for row in
                 json.loads((ROOT / "marker_mount_stability.json").read_text())}
    with Path("outputs/trial_review/decisions.csv").open(newline="") as stream:
        decisions = {row["stem"]: row for row in csv.DictReader(stream)}

    queue = []
    for row in replay_rows:
        clip = row["clip"]
        replay_dir = ROOT / "replays" / "dual_original" / clip / "wide_no_funnel"
        summary = json.loads((replay_dir / "summary.json").read_text())
        geometry = summary["socket_geometry"]
        if (geometry["inner_half_x_m"] != .009 or
                geometry["inner_half_y_m"] != .005 or
                geometry["funnel_depth_m"] != 0):
            raise ValueError(f"Unexpected geometry for {clip}: {geometry}")
        prefix_ok = (summary["source_motion_insertion_reached"] and
                     summary["source_prefix_joint_limits_ok"] and
                     summary["source_prefix_bad_collision_steps"] == 0)
        if not prefix_ok:
            continue
        steps = json.loads((replay_dir / "retargeted_replay.json").read_text())
        last = summary["initial_alignment_steps"] + summary["source_steps_to_first_success"]
        prefix = steps[:last + 1]
        force = max(step["metrics"]["max_wall_normal_force_n"] for step in prefix)
        penetration = max(step["metrics"]["max_wall_penetration_m"] for step in prefix) * 1000
        shift = stability.get(clip, {}).get("early_late_mount_translation_shift_mm")
        decision = decisions.get(clip, {})
        concerns = []
        if force > FORCE_SCREEN_N:
            concerns.append("contact_over_100_n")
        if shift is None:
            concerns.append("mount_stability_unavailable")
        elif shift > MOUNT_SHIFT_SCREEN_MM:
            concerns.append("mount_shift_over_4_mm")
        if row.get("approach_truncated"):
            concerns.append("approach_truncated")
        if decision.get("tracking") in ("bad", "questionable"):
            concerns.append("prior_tracking_concern")
        if decision.get("replay") in ("bad", "questionable"):
            concerns.append("prior_replay_concern")
        if decision.get("decision") == "reject":
            concerns.append("previously_rejected")
        tier = ("review_first" if not concerns else
                "review_with_caution" if force <= FORCE_SCREEN_N else
                "defer_high_contact")
        queue.append({
            "clip": clip, "tier": tier, "prior_decision": decision.get("decision", "unreviewed"),
            "prior_tracking": decision.get("tracking", "unknown"),
            "prior_replay": decision.get("replay", "unknown"),
            "strict_port_replay_entry": bool(row.get("strict_success")),
            "peak_contact_n_to_first_entry": round(force, 2),
            "peak_penetration_mm_to_first_entry": round(penetration, 3),
            "early_late_mount_shift_mm": round(shift, 2) if shift is not None else "",
            "approach_truncated": bool(row.get("approach_truncated")),
            "source_motion_steps_to_first_entry": summary["source_steps_to_first_success"] + 1,
            "concerns": ";".join(concerns),
            "source_track": row["source_track"],
            "replay_video": str(replay_dir / "replay.mp4"),
            "replay_summary": str(replay_dir / "summary.json"),
        })
    tier_order = {"review_first": 0, "review_with_caution": 1, "defer_high_contact": 2}
    queue.sort(key=lambda item: (tier_order[item["tier"]],
                                 not item["strict_port_replay_entry"],
                                 item["peak_contact_n_to_first_entry"], item["clip"]))
    output = ROOT / "wide_no_funnel_review_queue.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["review_order", *queue[0].keys()])
        writer.writeheader()
        for index, item in enumerate(queue, 1):
            writer.writerow({"review_order": index, **item})
    print(json.dumps({"candidates": len(queue),
                      "review_first": sum(item["tier"] == "review_first" for item in queue),
                      "review_with_caution": sum(item["tier"] == "review_with_caution"
                                                 for item in queue),
                      "defer_high_contact": sum(item["tier"] == "defer_high_contact"
                                                for item in queue),
                      "queue": str(output)}, indent=2))


if __name__ == "__main__":
    main()
