"""Rank additional widened-port candidates for visual review."""

import csv
import json
from pathlib import Path


def main() -> None:
    root = Path("outputs/recalibration")
    results = json.loads((root / "additional_wide_no_funnel_results.json").read_text())
    original = {(row["batch"], row["clip"]): row for row in
                json.loads((root / "replay_results.json").read_text())}
    queue = []
    for row in results:
        if not row.get("clean_prefix"):
            continue
        previous = original[(row["batch"], row["clip"])]
        force = row["peak_contact_n_to_first_entry"]
        concerns = []
        if force > 100:
            concerns.append("contact_over_100_n")
        if row["approach_truncated"]:
            concerns.append("approach_after_tracking_loss")
        if row["batch"] == "single_20mm":
            concerns.append("single_marker_no_cross_check")
        tier = ("defer_high_contact" if force > 100 else
                "review_with_caution" if row["approach_truncated"] else
                "review_next")
        replay = root / "replays" / row["batch"] / row["clip"] / "wide_no_funnel"
        queue.append({
            "batch": row["batch"], "clip": row["clip"], "tier": tier,
            "strict_port_replay_entry": bool(previous.get("strict_success")),
            "peak_contact_n_to_first_entry": round(force, 2),
            "approach_truncated": bool(row["approach_truncated"]),
            "concerns": ";".join(concerns),
            "source_track": row["source_track"],
            "replay_video": str(replay / "replay.mp4"),
            "replay_summary": str(replay / "summary.json"),
        })
    order = {"review_next": 0, "review_with_caution": 1, "defer_high_contact": 2}
    queue.sort(key=lambda row: (order[row["tier"]],
                                row["batch"] != "dual_minus3",
                                not row["strict_port_replay_entry"],
                                row["peak_contact_n_to_first_entry"], row["clip"]))
    output = root / "additional_wide_no_funnel_review_queue.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["review_order", *queue[0]])
        writer.writeheader()
        for index, row in enumerate(queue, 1):
            writer.writerow({"review_order": index, **row})
    print(json.dumps({"review_next": sum(row["tier"] == "review_next" for row in queue),
                      "review_with_caution": sum(row["tier"] == "review_with_caution"
                                                 for row in queue),
                      "defer_high_contact": sum(row["tier"] == "defer_high_contact"
                                                for row in queue)}, indent=2))


if __name__ == "__main__":
    main()
