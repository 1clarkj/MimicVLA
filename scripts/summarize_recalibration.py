"""Combine replay, fallback, contact, review, and dataset counts for the report."""

import csv
import json
from pathlib import Path


def main() -> None:
    root = Path("outputs/recalibration")
    replay = json.loads((root / "replay_summary.json").read_text())
    fallbacks = json.loads((root / "fallback_results.json").read_text())
    contacts = json.loads((root / "guided_contact_audit.json").read_text())
    with Path("outputs/trial_review/decisions.csv").open(newline="") as stream:
        decisions = list(csv.DictReader(stream))
    primary = replay["batches"]
    metrics = {
        "source_clips": sum(batch["clips"] for batch in primary.values()),
        "replayable": sum(batch["replayable"] for batch in primary.values()),
        "strict_replay_entries": sum(batch["strict"]["successes"] for batch in primary.values()),
        "guided_primary_technical_candidates": sum(
            batch["training"]["clean_source_prefixes"] for batch in primary.values()),
        "guided_marker0_fallback_candidates": sum(
            row.get("training_prefix_candidate", False) for row in fallbacks),
        "guided_candidates_total": len(contacts),
        "guided_candidates_under_100_n": sum(
            row["max_wall_force_n_before_success"] <= 100 for row in contacts),
        "reviewed_accept": sum(row["decision"] == "accept" for row in decisions),
        "reviewed_hold": sum(row["decision"] == "hold" for row in decisions),
        "unreviewed_restored_dual": primary["dual_original"]["clips"] - len(decisions),
        "datasets": {},
    }
    for name in ("usb_guided_reviewed_v2", "usb_guided_technical_v2",
                 "usb_guided_technical_lowforce_v3"):
        manifest = Path("outputs/datasets") / name / "capture_manifest.json"
        if manifest.exists():
            data = json.loads(manifest.read_text())
            metrics["datasets"][name] = {"episodes": data["episodes"],
                                         "frames": data["frames"]}
    (root / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
