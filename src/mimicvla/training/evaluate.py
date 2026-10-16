"""Run a trained LeRobot policy closed-loop in the 15 x 7 mm no-funnel sim.

Each rollout starts from the same aligned state as a dataset episode (the
source replay's alignment prefix is executed first), then the policy drives
the Panda until the tip centre is 12 mm deep or the step budget runs out.
By default only the held-out clips (-m mimicvla.dataset.splits) are used.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from mimicvla.dataset.splits import split_indices
from mimicvla.dataset.lerobot import TASK
from mimicvla.sim.env import PandaUSBEnv, SimConfig, insertion_success_from_metrics
from mimicvla.sim.scene import SocketGeometry


GEOMETRY = SocketGeometry(inner_half_x_m=.0075, inner_half_y_m=.0035)
SUCCESS_DEPTH_M = .012


def load_policy(checkpoint: Path, device: str, n_action_steps: int | None = None,
                num_steps: int | None = None):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.config.device = device
    if n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps
    if num_steps is not None:  # flow-matching denoising steps per chunk
        policy.config.num_steps = num_steps
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}})
    return policy, preprocessor, postprocessor


def policy_action(policy, preprocessor, postprocessor, obs: dict, device: str) -> np.ndarray:
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference

    frame = {key: obs[key] for key in ("observation.state", "observation.images.camera1",
                                       "observation.images.camera2")}
    batch = prepare_observation_for_inference(frame, torch.device(device), task=TASK,
                                              robot_type="franka_panda_usb")
    with torch.inference_mode():
        action = policy.select_action(preprocessor(batch))
    return postprocessor(action).squeeze(0).float().cpu().numpy().astype(float)


def alignment_actions(episode: dict) -> list[list[float]]:
    """Actions that move the arm from reset to the clip's start pose.

    Published start sets carry them inline; local ones read the saved replay.
    """
    if "alignment_actions" in episode:
        return episode["alignment_actions"]
    replay_dir = Path(episode.get("replay_dir") or
                      Path("outputs/recalibration/replays") / episode["batch"] /
                      episode["clip"] / "wide_no_funnel")
    replay = json.loads((replay_dir / "retargeted_replay.json").read_text())
    return [entry["action"] for entry in replay[:episode["first_source_step"]]]


def rollout(policy, preprocessor, postprocessor, episode: dict, args) -> dict:
    alignment = alignment_actions(episode)
    first = episode["first_source_step"]
    env = PandaUSBEnv(SimConfig(max_steps=10_000, geometry=GEOMETRY,
                                insertion_depth_m=SUCCESS_DEPTH_M))
    rng = np.random.default_rng(args.seed + episode["episode_index"])
    try:
        for action in alignment:
            env.step(np.asarray(action, dtype=float))
        # Optional extra start perturbation (world-frame metres) for robustness tests.
        if args.start_noise_mm > 0:
            offset = rng.uniform(-1, 1, 3) * args.start_noise_mm / 1000
            for _ in range(10):
                env.step(np.r_[offset / 10, 0, 0, 0])
            for _ in range(10):
                env.step(np.zeros(6))
        policy.reset()
        frames = []
        first_success = first_bad_collision = None
        max_depth = peak_force = 0.0
        chunk_ms = []
        info = env.metrics()
        for step in range(args.max_steps):
            obs = env.observation(render=True)
            if args.video:
                frames.append(np.concatenate([obs["observation.images.camera1"],
                                              obs["observation.images.camera2"]], axis=1))
            new_chunk = step % policy.config.n_action_steps == 0
            if new_chunk and args.device == "cuda":
                import torch
                torch.cuda.synchronize()
            started = time.perf_counter()
            action = policy_action(policy, preprocessor, postprocessor, obs, args.device)
            if new_chunk:  # this call ran the full VLM + denoising forward pass
                chunk_ms.append(1000 * (time.perf_counter() - started))
            _, _, info = env.step(action)
            max_depth = max(max_depth, info["axial_error_m"])
            peak_force = max(peak_force, info["max_wall_normal_force_n"])
            if info["bad_collision"] and first_bad_collision is None:
                first_bad_collision = step
            if info["insertion_success"]:
                first_success = step
                if not args.continue_after_success:
                    break
        if args.video:
            video_path = args.output_dir / f"episode_{episode['episode_index']:03d}.mp4"
            with imageio.get_writer(video_path, fps=env.config.control_hz,
                                    codec="libx264", quality=8) as writer:
                for frame in frames:
                    writer.append_data(frame)
        return {
            "episode_index": episode["episode_index"], "clip": episode["clip"],
            "success": first_success is not None,
            "steps_to_success": first_success,
            "demo_steps_to_12mm": (None if episode.get("first_12mm_step") is None
                                   else episode["first_12mm_step"] - first),
            "first_bad_collision_step": first_bad_collision,
            "max_depth_mm": 1000 * max_depth,
            "final_depth_mm": 1000 * info["axial_error_m"],
            "final_lateral_mm": [1000 * info["socket_lateral_x_m"],
                                 1000 * info["socket_lateral_y_m"]],
            "final_orientation_error_deg": float(np.rad2deg(info["orientation_error_rad"])),
            "final_tip_in_opening_7mm": insertion_success_from_metrics(
                info, required_depth_m=.007, geometry=GEOMETRY),
            "peak_wall_force_n": peak_force,
            "median_chunk_ms": float(np.median(chunk_ms)) if chunk_ms else None,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path,
                        help="e.g. outputs/train/RUN/checkpoints/last/pretrained_model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path("outputs/datasets/usb_15x7_no_funnel_full_v2"),
                        help="dataset whose episodes supply the start states")
    parser.add_argument("--split", choices=["holdout", "train", "all"], default="holdout")
    parser.add_argument("--episodes", type=int, nargs="*",
                        help="explicit dataset episode indices (overrides --split)")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--n-action-steps", type=int,
                        help="actions executed per predicted chunk (default: training value)")
    parser.add_argument("--num-steps", type=int,
                        help="flow-matching denoising steps per chunk (default: training value, 10)")
    parser.add_argument("--start-noise-mm", type=float, default=0.0,
                        help="random world-frame start offset added after alignment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-video", dest="video", action="store_false")
    parser.add_argument("--continue-after-success", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.dataset_dir / "capture_manifest.json").read_text())
    episodes = manifest["episodes_manifest"]
    if args.episodes is not None:
        episodes = [episodes[index] for index in args.episodes]
    elif args.split != "all":
        episodes = [episodes[index] for index in split_indices(args.dataset_dir)[args.split]]
    import torch

    torch.manual_seed(args.seed)
    policy, preprocessor, postprocessor = load_policy(args.checkpoint, args.device,
                                                      args.n_action_steps, args.num_steps)

    results = []
    for episode in episodes:
        result = rollout(policy, preprocessor, postprocessor, episode, args)
        results.append(result)
        print(json.dumps(result), flush=True)
        successes = sum(row["success"] for row in results)
        summary = {"checkpoint": str(args.checkpoint), "dataset": str(args.dataset_dir),
                   "split": "explicit" if args.episodes is not None else args.split,
                   "success_depth_mm": 12,
                   "start_noise_mm": args.start_noise_mm, "max_steps": args.max_steps,
                   "n_action_steps": policy.config.n_action_steps,
                   "num_steps": policy.config.num_steps,
                   "rollouts": len(results), "successes": successes,
                   "success_rate": successes / len(results),
                   "bad_collision_rollouts": sum(row["first_bad_collision_step"] is not None
                                                 for row in results),
                   "results": results}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{summary['successes']}/{summary['rollouts']} succeeded "
          f"({100 * summary['success_rate']:.0f}%), "
          f"{summary['bad_collision_rollouts']} with deep collisions")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(Path("outputs/hf_cache").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
