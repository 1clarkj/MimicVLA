"""Review queues for the widened, funnel-free port.

Runs three steps in order: a two-marker mount-stability check (a review flag,
not proof that a marker moved), the queue for the restored-mount two-marker
batch, and the queue for the single-marker and earlier two-marker clips
(replayed by `mimicvla.dataset.replay_additional`).

    python -m mimicvla.dataset.curate
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mimicvla.vision.transforms import inverse


ROOT = Path("outputs/recalibration")
FORCE_SCREEN_N = 100.0  # Dataset triage heuristic, not a hardware safety limit.
MOUNT_SHIFT_SCREEN_MM = 4.0


def mount_stability() -> None:
    root = Path("outputs/recalibration")
    rows = []
    for path in sorted(Path("outputs/dual_marker_original_mount/dual").glob(
            "*/tracking/trajectory.jsonl")):
        translations = []
        for line in path.read_text().splitlines():
            record = json.loads(line)
            markers = record["markers"]
            if "0" not in markers or "10" not in markers:
                continue
            marker0 = np.asarray(markers["0"]["T_camera_marker"])
            marker10 = np.asarray(markers["10"]["T_camera_marker"])
            translations.append((inverse(marker0) @ marker10)[:3, 3] * 1000)
        n = len(translations)
        row = {"clip": path.parent.parent.name, "co_visible_frames": n,
               "early_late_mount_translation_shift_mm": None,
               "possible_mount_change_or_pose_error": False}
        if n >= 24:
            third = n // 3
            translation = np.asarray(translations)
            early = np.median(translation[:third], axis=0)
            late = np.median(translation[-third:], axis=0)
            shift = float(np.linalg.norm(late - early))
            row.update(early_late_mount_translation_shift_mm=shift,
                       possible_mount_change_or_pose_error=shift > 4)
        rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    (root / "marker_mount_stability.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (root / "marker_mount_stability.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assessed = [row for row in rows if row["early_late_mount_translation_shift_mm"] is not None]
    print(f"Assessed {len(assessed)} clips; {sum(row['possible_mount_change_or_pose_error'] for row in assessed)} exceed 4 mm."
          " This is a tracking/mount review flag, not proof that a marker moved.")



def two_marker_queue() -> None:
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



def additional_queue() -> None:
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



def main() -> None:
    mount_stability()
    two_marker_queue()
    additional_queue()


if __name__ == "__main__":
    main()
