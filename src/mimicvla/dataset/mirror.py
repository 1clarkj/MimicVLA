"""Add left-right mirrored copies of the 72 reviewed 15 x 7 mm demonstrations.

Data augmentation for the scaling curve. Each human tip trajectory is
reflected across the socket's local Y-Z plane (world left-right; the
rectangular opening is symmetric about it), re-retargeted with Mink IK and
re-rendered. Images are therefore rendered from the mirrored robot motion,
not flipped: the fixed camera and the wrist camera (mounted 45 mm to one side
of the plug) are not symmetric, so flipping pixels would not match any real
robot state. The dataset holds the original v2 episodes followed by every
mirrored copy that passes the v2 gate. Mirrored copies keep their source
(batch, clip) key, so holdout_split.py also holds out the mirrors of the
holdout clips.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mimicvla.dataset.export_review import CAMERA_SETUP, FEATURES
from mimicvla.dataset.lowpass import GEOMETRY, REPLAY_ROOT, SOURCE_DATASET
from mimicvla.dataset.splits import HOLDOUT_CLIPS
from mimicvla.dataset.lerobot import TASK
from mimicvla.retarget import replay_recording
from mimicvla.sim.env import PandaUSBEnv, SimConfig, insertion_success_from_metrics


def mirrored_replay(episode: dict) -> tuple[Path, int]:
    clip_dir = REPLAY_ROOT / episode["batch"] / episode["clip"]
    source = json.loads((clip_dir / "wide_no_funnel" / "summary.json").read_text())
    replay_dir = clip_dir / "mirror_x_15x7"
    summary = replay_recording(
        Path(source["source"]), replay_dir,
        start_frame=source["source_frames"][0], end_frame=source["source_frames"][1],
        complete_segment=True, marker_tip=np.asarray(source["T_marker0_usb_tip"]),
        sim_config=SimConfig(max_steps=1000, geometry=GEOMETRY),
        write_video=False, mirror_socket_x=True)
    return replay_dir, summary["initial_alignment_steps"]


def execute(replay: list[dict], first: int) -> tuple[list[dict], dict | None, str | None]:
    """Re-run replay actions, recording frames after the alignment prefix (v2 gate)."""
    env = PandaUSBEnv(SimConfig(max_steps=1000, geometry=GEOMETRY, insertion_depth_m=.007))
    rows, failure, stats = [], None, {"first_7mm_step": None, "first_12mm_step": None,
                                      "peak_wall_force_n_full_replay": 0.0}
    try:
        for index, entry in enumerate(replay):
            action = np.asarray(entry["action"], dtype=float)
            if index >= first:
                obs = env.observation(render=True)
                rows.append({"observation.state": obs["observation.state"],
                             "observation.images.camera1": obs["observation.images.camera1"],
                             "observation.images.camera2": obs["observation.images.camera2"],
                             "action": action.astype(np.float32), "task": TASK})
            _, _, info = env.step(action)
            stats["peak_wall_force_n_full_replay"] = max(
                stats["peak_wall_force_n_full_replay"], info["max_wall_normal_force_n"])
            joints = env.observation(render=False)["observation.state"][:7]
            limits = env.model.jnt_range[:7]
            if info["bad_collision"]:
                return rows, None, f"deep collision at step {index}"
            if np.any(joints < limits[:, 0] - 1e-4) or np.any(joints > limits[:, 1] + 1e-4):
                return rows, None, f"joint limit at step {index}"
            if index >= first and info["insertion_success"] and stats["first_7mm_step"] is None:
                stats["first_7mm_step"] = index
            if (index >= first and stats["first_12mm_step"] is None and
                    insertion_success_from_metrics(info, required_depth_m=.012,
                                                   geometry=GEOMETRY)):
                stats["first_12mm_step"] = index
        if info["axial_error_m"] < .012:
            failure = f"final depth {1000 * info['axial_error_m']:.1f} mm"
        stats["final_depth_m"] = info["axial_error_m"]
        return rows, stats, failure
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/datasets/usb_15x7_no_funnel_mirror_v1"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty dataset directory: {args.output_dir}")
    source_manifest = json.loads((SOURCE_DATASET / "capture_manifest.json").read_text())
    episodes = source_manifest["episodes_manifest"]
    if len(episodes) != 72:
        raise ValueError("Expected the 72 reviewed v2 demonstrations")

    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=f"mimicvla/{args.output_dir.name}", fps=20, features=FEATURES,
        root=args.output_dir, robot_type="franka_panda_usb",
        use_videos=True, video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=23, preset="veryfast", g=20))
    kept, rejected = [], []
    try:
        for mirrored in (False, True):
            for episode in episodes:
                key = (episode["batch"], episode["clip"])
                if mirrored:
                    try:
                        replay_dir, first = mirrored_replay(episode)
                    except ValueError as error:
                        rejected.append({"batch": key[0], "clip": key[1], "reason": str(error)})
                        continue
                else:
                    replay_dir = Path(episode.get("replay_dir") or
                                      REPLAY_ROOT / key[0] / key[1] / "wide_no_funnel")
                    first = episode["first_source_step"]
                replay = json.loads((replay_dir / "retargeted_replay.json").read_text())
                rows, stats, failure = execute(replay, first)
                if failure is not None:
                    if not mirrored:
                        raise RuntimeError(f"v2 episode {key} no longer passes: {failure}")
                    rejected.append({"batch": key[0], "clip": key[1], "reason": failure})
                    print(episode["clip"], "mirror rejected:", failure, flush=True)
                    continue
                for row in rows:
                    dataset.add_frame(row)
                dataset.save_episode()
                kept.append({
                    "episode_index": len(kept), "batch": key[0], "clip": key[1],
                    "mirrored": mirrored,
                    "split": "holdout" if key in HOLDOUT_CLIPS else "train",
                    "v2_episode_index": episode["episode_index"],
                    "source_track": episode["source_track"], "replay_dir": str(replay_dir),
                    "calibration_epoch": episode["calibration_epoch"],
                    "human_review_decision": "accept", "frames": len(rows),
                    "first_source_step": first, "last_source_step": len(replay) - 1,
                    **stats})
                print(episode["clip"], "mirror" if mirrored else "orig", len(rows), "frames",
                      flush=True)
    finally:
        dataset.finalize()

    manifest = {"selection": "the 72 reviewed v2 demonstrations plus their left-right "
                             "mirrored, re-retargeted copies that pass the v2 gate",
                "mirror": "socket-frame reflection p -> M p, R -> M R M, M = diag(-1, 1, 1); "
                          "socket local X is world -Y",
                "episodes": len(kept), "frames": sum(row["frames"] for row in kept),
                "mirrored_episodes": sum(row["mirrored"] for row in kept),
                "holdout_episodes": [row["episode_index"] for row in kept
                                     if row["split"] == "holdout"],
                "geometry": vars(GEOMETRY), "camera_setup": CAMERA_SETUP,
                "debug_sites_visible": False, "source_dataset": str(SOURCE_DATASET),
                "gate": "no flagged deep collision or joint-limit failure over the whole "
                        "replay, and finishes at least 12 mm deep",
                "rejected": rejected, "episodes_manifest": kept}
    (args.output_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"episodes": manifest["episodes"], "frames": manifest["frames"],
                      "mirrored": manifest["mirrored_episodes"], "rejected": len(rejected)}))


if __name__ == "__main__":
    main()
