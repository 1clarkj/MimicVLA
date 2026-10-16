"""Replay reviewed demonstrations against no-funnel throat sizes at a chosen depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mimicvla.sim.env import PandaUSBEnv, SimConfig
from mimicvla.sim.scene import SocketGeometry


def evaluate_clip(root: Path, episode: dict, geometry: SocketGeometry,
                  depth_m: float) -> dict:
    replay_dir = root / "replays" / episode["batch"] / episode["clip"] / "wide_no_funnel"
    replay = json.loads((replay_dir / "retargeted_replay.json").read_text())
    summary = json.loads((replay_dir / "summary.json").read_text())
    first_source = summary["initial_alignment_steps"]
    env = PandaUSBEnv(SimConfig(max_steps=1000, geometry=geometry,
                                insertion_depth_m=depth_m))
    first_success = None
    failure = None
    peak_force_n = 0.0
    try:
        for index, row in enumerate(replay):
            _, _, metrics = env.step(np.asarray(row["action"], dtype=float))
            peak_force_n = max(peak_force_n, metrics["max_wall_normal_force_n"])
            if metrics["bad_collision"]:
                failure = "bad_collision"
                break
            joints = env.observation(render=False)["observation.state"][:7]
            limits = env.model.jnt_range[:7]
            if np.any(joints < limits[:, 0] - 1e-4) or np.any(joints > limits[:, 1] + 1e-4):
                failure = "joint_limit"
                break
            if index >= first_source and metrics["insertion_success"]:
                first_success = index
                break
        if first_success is None and failure is None:
            failure = "no_entry_by_end_of_source_motion"
    finally:
        env.close()
    return {"batch": episode["batch"], "clip": episode["clip"],
            "valid": first_success is not None, "first_success_step": first_success,
            "failure": failure, "peak_force_n_to_stop": peak_force_n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-mm", type=float, default=7)
    parser.add_argument("--size", nargs="+", default=["18x10", "16x9", "16x8",
                                                   "15x8", "15x7", "14x7", "13.4x5.9"],
                        help="Full throat width x height in mm, with no funnel")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/recalibration/no_funnel_7mm_port_sweep.json"))
    parser.add_argument("--limit", type=int, help="Only for quick smoke checks")
    args = parser.parse_args()
    if args.depth_mm <= 0:
        parser.error("--depth-mm must be positive")
    manifest = json.loads(Path("outputs/datasets/usb_wide_no_funnel_reviewed_all_v2/"
                               "capture_manifest.json").read_text())
    episodes = manifest["episodes_manifest"][:args.limit]
    if len(episodes) != (args.limit or manifest["episodes"]):
        raise ValueError("Dataset manifest episode count is inconsistent")
    root = Path("outputs/recalibration")
    result = {"source_manifest": "outputs/datasets/usb_wide_no_funnel_reviewed_all_v2/"
                                 "capture_manifest.json",
              "criterion": f"first {args.depth_mm:g} mm tip-center entry during source motion, "
                           "no flagged collision or joint-limit failure through entry",
              "depth_mm": args.depth_mm, "source_episodes": len(episodes), "sizes": []}
    for size in args.size:
        try:
            width_mm, height_mm = map(float, size.lower().split("x"))
            geometry = SocketGeometry(inner_half_x_m=width_mm / 2000,
                                      inner_half_y_m=height_mm / 2000,
                                      mouth_half_x_m=width_mm / 2000,
                                      mouth_half_y_m=height_mm / 2000)
        except (ValueError, TypeError) as error:
            parser.error(f"Invalid --size {size!r}: {error}")
        rows = []
        for episode in episodes:
            rows.append(evaluate_clip(root, episode, geometry, args.depth_mm / 1000))
        entry = {"width_mm": width_mm, "height_mm": height_mm,
                 "opening_area_mm2": width_mm * height_mm,
                 "valid_episodes": sum(row["valid"] for row in rows),
                 "episodes": rows}
        result["sizes"].append(entry)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{size}: {entry['valid_episodes']}/{len(episodes)}", flush=True)


if __name__ == "__main__":
    main()
