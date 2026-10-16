"""Render guided Panda replays for the visual trial-review dashboard."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mimicvla.retarget import replay_recording
from mimicvla.sim.env import SimConfig
from mimicvla.sim.scene import SocketGeometry, TRAINING_SOCKET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="dual_original")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--geometry", choices=("training", "wide_no_funnel"),
                        default="training")
    parser.add_argument("--review-queue", action="store_true",
                        help="render low-contact candidates from the no-funnel queue")
    args = parser.parse_args()
    if args.fallback and args.geometry != "training":
        parser.error("marker-0 fallback was only replayed with training geometry")
    if args.review_queue and args.geometry != "wide_no_funnel":
        parser.error("--review-queue requires wide_no_funnel geometry")
    root = Path("outputs/recalibration")
    source_file = root / ("fallback_results.json" if args.fallback else "replay_results.json")
    rows = [row for row in json.loads(source_file.read_text())
            if row["batch"] == args.batch and row["status"] == "replayable"
            and (not args.fallback or row.get("training_prefix_candidate"))]
    if args.review_queue:
        queue_file = ("wide_no_funnel_review_queue.csv" if args.batch == "dual_original"
                      else "additional_wide_no_funnel_review_queue.csv")
        with (root / queue_file).open(newline="") as stream:
            clips = [item["clip"] for item in csv.DictReader(stream)
                     if item["tier"] != "defer_high_contact"
                     and item.get("batch", args.batch) == args.batch]
        by_clip = {row["clip"]: row for row in rows}
        rows = [by_clip[clip] for clip in clips]
    epochs = json.loads((root / "candidate_mounts.json").read_text())["batches"][args.batch]["epochs"]
    for row in rows[:args.limit]:
        folder = root / "replays" / args.batch / row["clip"]
        if args.fallback:
            folder /= "marker0_fallback"
        output = folder / args.geometry
        if (output / "replay.mp4").exists():
            continue
        epoch = next(epoch for epoch in epochs
                     if epoch["first_clip"] <= row["clip"] <= epoch["last_clip"])
        replay_recording(folder / "trajectory_filled.jsonl", output,
                         start_frame=row["start_frame"], end_frame=row["end_frame"],
                         complete_segment=True,
                         marker_tip=np.asarray(epoch["T_marker0_usb_tip"]),
                         sim_config=SimConfig(max_steps=1000, geometry=(
                             TRAINING_SOCKET if args.geometry == "training" else
                             SocketGeometry(inner_half_x_m=.009, inner_half_y_m=.005,
                                            mouth_half_x_m=.009, mouth_half_y_m=.005))),
                         write_video=True)
        print(row["clip"], flush=True)


if __name__ == "__main__":
    main()
