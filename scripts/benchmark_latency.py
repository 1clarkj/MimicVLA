"""Time one SmolVLA action-chunk prediction per denoising-step setting.

Run on an otherwise idle GPU. Observations come from real start states (the
first rendered frame after each dataset episode's alignment prefix). Each
timed call resets the action queue, so it runs the full VLM prefix plus
``num_steps`` flow-matching denoising passes and returns a 50-step chunk.

    PYTHONPATH=scripts MUJOCO_GL=egl python scripts/benchmark_latency.py CHECKPOINT \
        --steps 1 2 3 5 10 --output outputs/latency/lowpass_20k.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from evaluate_policy import GEOMETRY, load_policy, policy_action
from mimicvla.sim.env import PandaUSBEnv, SimConfig


def start_observations(dataset_dir: Path, count: int) -> list[dict]:
    episodes = json.loads((dataset_dir / "capture_manifest.json").read_text())["episodes_manifest"]
    observations = []
    for episode in episodes[:count]:
        replay_dir = (Path("outputs/recalibration/replays") / episode["batch"] /
                      episode["clip"] / "wide_no_funnel")
        replay = json.loads((replay_dir / "retargeted_replay.json").read_text())
        env = PandaUSBEnv(SimConfig(max_steps=10_000, geometry=GEOMETRY))
        for entry in replay[:episode["first_source_step"]]:
            env.step(np.asarray(entry["action"], dtype=float))
        observations.append(env.observation(render=True))
        env.close()
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path("outputs/datasets/usb_15x7_no_funnel_full_v2"))
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    parser.add_argument("--observations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    observations = start_observations(args.dataset_dir, args.observations)
    policy, pre, post = load_policy(args.checkpoint, args.device, 10)
    rows = []
    for num_steps in args.steps:
        policy.config.num_steps = num_steps
        for obs in observations[:3]:  # warm-up
            policy.reset()
            policy_action(policy, pre, post, obs, args.device)
        times = []
        for _ in range(args.repeats):
            for obs in observations:
                policy.reset()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                policy_action(policy, pre, post, obs, args.device)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                times.append(1000 * (time.perf_counter() - started))
        rows.append({"num_steps": num_steps, "median_ms": float(np.median(times)),
                     "p90_ms": float(np.percentile(times, 90)), "calls": len(times)})
        print(json.dumps(rows[-1]), flush=True)
    device_name = (torch.cuda.get_device_name() if args.device == "cuda"
                   else f"CPU ({os.cpu_count()} threads)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checkpoint": str(args.checkpoint), "device": device_name,
                                       "chunk_size": policy.config.chunk_size,
                                       "n_action_steps": policy.config.n_action_steps,
                                       "results": rows}, indent=2) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(Path("outputs/hf_cache").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
