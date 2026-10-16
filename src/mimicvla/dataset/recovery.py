"""DART-style recovery data from the reviewed 15 x 7 mm human demonstrations.

For each training clip (held-out clips are excluded entirely), the human
replay is re-executed with a smooth random offset added to the *executed*
command. Every frame is labelled with the clean command that steers the
controller target back to the human's target for that step, so the policy
sees off-path states paired with corrective actions. ``--correction-gain``
sets the fraction of the current offset each label removes in one step; 1.0
(v1) snaps straight back and makes rotation labels much larger than the human's,
smaller values return over several steps. Copy 0 has no noise and
reproduces the v2 episode. The offset fades out near the port mouth so the
perturbation never pushes the plug into the walls.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from mimicvla.dataset.splits import HOLDOUT_CLIPS
from mimicvla.dataset.lerobot import CAMERA_SETUP, FEATURES, TASK
from mimicvla.sim.env import PandaUSBEnv, SimConfig
from mimicvla.sim.scene import SocketGeometry


GEOMETRY = SocketGeometry(inner_half_x_m=.0075, inner_half_y_m=.0035)
DEFAULT_SOURCE = Path("outputs/datasets/usb_15x7")
MAX_STEP_M = .003
MAX_STEP_RAD = np.deg2rad(2)


def command_to(env: PandaUSBEnv, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """World-frame displacement from the controller target, capped like the replays."""
    delta_position = position - env.controller.target_position
    delta_rotation = Rotation.from_matrix(
        rotation @ env.controller.target_rotation.T).as_rotvec()
    distance = np.linalg.norm(delta_position)
    angle = np.linalg.norm(delta_rotation)
    if distance > MAX_STEP_M:
        delta_position *= MAX_STEP_M / distance
    if angle > MAX_STEP_RAD:
        delta_rotation *= MAX_STEP_RAD / angle
    return np.r_[delta_position, delta_rotation]


def noise_scale(depth_m: float, fade_start_m: float, fade_end_m: float) -> float:
    """1 well before the mouth, 0 from fade_end_m before the mouth onwards."""
    return float(np.clip((-depth_m - fade_end_m) / (fade_start_m - fade_end_m), 0, 1))


def capture(job: dict) -> dict:
    episode, copy, config = job["episode"], job["copy"], job["config"]
    replay_dir = Path(episode["replay_dir"])
    replay = json.loads((replay_dir / "retargeted_replay.json").read_text())
    first = episode["first_source_step"]
    rng = np.random.default_rng([config["seed"], episode["episode_index"], copy])
    sigma = 0.0 if copy == 0 else 1.0
    theta = config["theta"]
    # Per-step OU kick sized so the stationary offset std matches the request.
    kick_m = sigma * config["position_std_m"] * np.sqrt(2 * theta - theta ** 2)
    kick_rad = sigma * config["rotation_std_rad"] * np.sqrt(2 * theta - theta ** 2)
    env = PandaUSBEnv(SimConfig(max_steps=10_000, geometry=GEOMETRY,
                                insertion_depth_m=.012))
    frames = []
    offset = np.zeros(6)
    offsets = []
    previous = np.zeros(6)
    keep = 1 - config["correction_gain"]
    failure = None
    try:
        for entry in replay[:first]:
            env.step(np.asarray(entry["action"], dtype=float))
        info = env.metrics()
        for entry in replay[first:]:
            target = np.asarray(entry["T_world_ee_target"], dtype=float)
            # The controller target carries the previous offset; remove only part of it.
            label = command_to(env, target[:3, 3] + keep * previous[:3],
                               Rotation.from_rotvec(keep * previous[3:]).as_matrix() @
                               target[:3, :3])
            offset = (1 - theta) * offset + np.r_[rng.normal(0, kick_m, 3),
                                                  rng.normal(0, kick_rad, 3)]
            applied = offset * noise_scale(info["axial_error_m"], config["fade_start_m"],
                                           config["fade_end_m"])
            offsets.append(applied)
            previous = applied
            executed = command_to(env, target[:3, 3] + applied[:3],
                                  Rotation.from_rotvec(applied[3:]).as_matrix() @ target[:3, :3])
            obs = env.observation(render=True)
            frames.append({"observation.state": obs["observation.state"],
                           "observation.images.camera1": obs["observation.images.camera1"],
                           "observation.images.camera2": obs["observation.images.camera2"],
                           "action": label.astype(np.float32), "task": TASK})
            _, _, info = env.step(executed)
            joints = env.observation(render=False)["observation.state"][:7]
            limits = env.model.jnt_range[:7]
            if info["bad_collision"]:
                failure = f"deep collision at step {len(frames)}"
                break
            if np.any(joints < limits[:, 0] - 1e-4) or np.any(joints > limits[:, 1] + 1e-4):
                failure = f"joint limit at step {len(frames)}"
                break
        if failure is None and info["axial_error_m"] < .012:
            failure = f"final depth {1000 * info['axial_error_m']:.1f} mm"
    finally:
        env.close()
    offsets = np.asarray(offsets)
    return {"episode": episode, "copy": copy, "failure": failure,
            "frames": frames if failure is None else [],
            "final_depth_m": info["axial_error_m"],
            "rms_offset_mm": float(1000 * np.sqrt((offsets[:, :3] ** 2).sum(1).mean())),
            "max_offset_mm": float(1000 * np.linalg.norm(offsets[:, :3], axis=1).max()),
            "rms_offset_deg": float(np.rad2deg(np.sqrt((offsets[:, 3:] ** 2).sum(1).mean())))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=3,
                        help="copies per clip, including the clean copy 0")
    parser.add_argument("--position-std-mm", type=float, default=4.0)
    parser.add_argument("--rotation-std-deg", type=float, default=3.0)
    parser.add_argument("--theta", type=float, default=.05,
                        help="OU mean reversion per 20 Hz step (0.05 = 1 s time constant)")
    parser.add_argument("--fade-start-mm", type=float, default=30.0)
    parser.add_argument("--fade-end-mm", type=float, default=5.0)
    parser.add_argument("--correction-gain", type=float, default=1.0,
                        help="fraction of the offset each label removes (v1 used 1.0)")
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE,
                        help="dataset whose episodes (and replays) to perturb, e.g. the "
                             "low-pass dataset")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-id", default="usb_15x7_recovery")
    args = parser.parse_args()
    output_dir = Path("outputs/datasets") / args.dataset_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty dataset directory: {output_dir}")
    source = json.loads((args.source_dataset / "capture_manifest.json").read_text())
    episodes = [episode for episode in source["episodes_manifest"]
                if episode["clip"] not in HOLDOUT_CLIPS]
    config = {"seed": args.seed, "theta": args.theta,
              "correction_gain": args.correction_gain,
              "position_std_m": args.position_std_mm / 1000,
              "rotation_std_rad": np.deg2rad(args.rotation_std_deg),
              "fade_start_m": args.fade_start_mm / 1000, "fade_end_m": args.fade_end_mm / 1000}
    jobs = [{"episode": episode, "copy": copy, "config": config}
            for episode in episodes for copy in range(args.copies)]

    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=f"mimicvla/{args.dataset_id}", fps=20, features=FEATURES,
        root=output_dir, robot_type="franka_panda_usb",
        use_videos=True, video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=23, preset="veryfast", g=20))
    kept, rejected = [], []
    try:
        with ProcessPoolExecutor(args.workers) as pool:
            for result in pool.map(capture, jobs, chunksize=1):
                episode = result["episode"]
                row = {"clip": episode["clip"],
                       "copy": result["copy"], "source_episode_index": episode["episode_index"],
                       "rms_offset_mm": result["rms_offset_mm"],
                       "max_offset_mm": result["max_offset_mm"],
                       "rms_offset_deg": result["rms_offset_deg"]}
                if result["failure"] is not None:
                    rejected.append({**row, "reason": result["failure"]})
                    print(episode["clip"], result["copy"], "rejected:", result["failure"],
                          flush=True)
                    continue
                for frame in result["frames"]:
                    dataset.add_frame(frame)
                dataset.save_episode()
                kept.append({"episode_index": len(kept), **row, "split": "train",
                             "frames": len(result["frames"]),
                             "first_source_step": episode["first_source_step"],
                             "final_depth_m": result["final_depth_m"],
                             "human_review_decision": "accept"})
                print(episode["clip"], result["copy"], len(result["frames"]), "frames",
                      f"rms offset {result['rms_offset_mm']:.1f} mm", flush=True)
    finally:
        dataset.finalize()

    manifest = {"selection": "the non-held-out clips of the source dataset; copy 0 clean, "
                             "other copies with DART-style execution noise",
                "noise": {"type": "Ornstein-Uhlenbeck offset on the executed target",
                          "position_std_mm": args.position_std_mm,
                          "rotation_std_deg": args.rotation_std_deg, "theta": args.theta,
                          "fade_start_mm": args.fade_start_mm,
                          "fade_end_mm": args.fade_end_mm, "seed": args.seed,
                          "correction_gain": args.correction_gain},
                "labels": "clean command from the controller target to the human target "
                          "for that step, capped at 3 mm and 2 deg per step",
                "episodes": len(kept), "frames": sum(row["frames"] for row in kept),
                "holdout_excluded": sorted(HOLDOUT_CLIPS),
                "geometry": vars(GEOMETRY), "camera_setup": CAMERA_SETUP,
                "source_dataset": str(args.source_dataset),
                "gate": "no flagged deep collision or joint-limit failure, and finishes "
                        "at least 12 mm deep (the build-dataset gate)",
                "rejected": rejected, "episodes_manifest": kept}
    (output_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"episodes": manifest["episodes"], "frames": manifest["frames"],
                      "rejected": len(rejected)}))


if __name__ == "__main__":
    main()
