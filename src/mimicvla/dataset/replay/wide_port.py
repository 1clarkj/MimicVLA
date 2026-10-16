"""Replay and render clips on the widened 18 x 10 mm port for the review dashboard.

    python -m mimicvla.dataset.replay.wide_port [--batch dual_original] [--review-queue]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mimicvla.retarget import replay_recording
from mimicvla.sim.env import SimConfig
from mimicvla.sim.scene import SocketGeometry

WIDE_PORT = SocketGeometry(inner_half_x_m=.009, inner_half_y_m=.005)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="dual_original")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--review-queue", action="store_true",
                        help="render low-contact candidates from the no-funnel queue")
    args = parser.parse_args()
    root = Path("outputs/recalibration")
    rows = [row for row in json.loads((root / "replay_results.json").read_text())
            if row["batch"] == args.batch and row["status"] == "replayable"]
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
        output = folder / "wide_no_funnel"
        if (output / "replay.mp4").exists():
            continue
        epoch = next(epoch for epoch in epochs
                     if epoch["first_clip"] <= row["clip"] <= epoch["last_clip"])
        replay_recording(folder / "trajectory_filled.jsonl", output,
                         start_frame=row["start_frame"], end_frame=row["end_frame"],
                         complete_segment=True,
                         marker_tip=np.asarray(epoch["T_marker0_usb_tip"]),
                         sim_config=SimConfig(max_steps=1000, geometry=WIDE_PORT),
                         write_video=True)
        print(row["clip"], flush=True)


if __name__ == "__main__":
    main()
