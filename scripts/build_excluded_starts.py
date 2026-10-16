"""Build the excluded-start evaluation set.

Every recorded clip whose widened-port replay reaches the aligned start pose,
minus the 72 clips of the training dataset. Only the start state is used:
`evaluate_policy.py` replays the clip's alignment prefix, then hands control
to the policy. No model trains on these clips.
"""
import argparse
import csv
import json
from pathlib import Path

REPLAYS = Path("outputs/recalibration/replays")
DATASET = Path("outputs/datasets/usb_15x7_no_funnel_full_v2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_sets/excluded_starts_v1"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output_dir / "capture_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"{manifest_path} exists; pass --overwrite to replace it")

    dataset = json.loads((DATASET / "capture_manifest.json").read_text())["episodes_manifest"]
    in_dataset = {(e["batch"], e["clip"]) for e in dataset}
    with Path("outputs/trial_review/decisions.csv").open(newline="") as stream:
        review = {row["stem"]: row["decision"] for row in csv.DictReader(stream)}
    episodes = []
    for replay_dir in sorted(REPLAYS.glob("*/*/wide_no_funnel")):
        key = (replay_dir.parts[-3], replay_dir.parts[-2])
        summary_path = replay_dir / "summary.json"
        if key in in_dataset or not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if not summary.get("alignment_reached"):
            continue
        episodes.append({"episode_index": len(episodes), "batch": key[0], "clip": key[1],
                         "replay_dir": str(replay_dir),
                         "first_source_step": summary["initial_alignment_steps"],
                         "human_review_decision": review.get(key[1], "not_in_review")})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "selection": "start states of recorded clips that are not in the 72-episode 15x7 "
                     "dataset and whose replay reached the aligned start",
        "episodes": len(episodes), "episodes_manifest": episodes}, indent=2) + "\n")
    print(f"{len(episodes)} excluded starts -> {manifest_path}")


if __name__ == "__main__":
    main()
