"""Re-retarget the 72 reviewed 15 x 7 mm demonstrations with low-pass tracking.

Same human clips, segments and marker mounts as usb_15x7_no_funnel_full_v2;
the only change is a zero-phase low-pass on the tracked tip pose (position and
orientation) before retargeting. Each smoothed replay is re-executed in the
15 x 7 mm port and kept only if it passes the v2 gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from export_recalibrated_lerobot import CAMERA_SETUP, FEATURES
from holdout_split import HOLDOUT_CLIPS
from mimicvla.data.sim_export import TASK
from mimicvla.retarget import replay_recording
from mimicvla.sim.env import PandaUSBEnv, SimConfig, insertion_success_from_metrics
from mimicvla.sim.scene import SocketGeometry


GEOMETRY = SocketGeometry(inner_half_x_m=.0075, inner_half_y_m=.0035,
                          mouth_half_x_m=.0075, mouth_half_y_m=.0035)
SOURCE_DATASET = Path("outputs/datasets/usb_15x7_no_funnel_full_v2")
REPLAY_ROOT = Path("outputs/recalibration/replays")


def smoothed_replay(episode: dict, lowpass_hz: float) -> tuple[Path, dict]:
    clip_dir = REPLAY_ROOT / episode["batch"] / episode["clip"]
    source = json.loads((clip_dir / "wide_no_funnel" / "summary.json").read_text())
    replay_dir = clip_dir / f"lowpass_{lowpass_hz:g}hz_15x7"
    summary = replay_recording(
        Path(source["source"]), replay_dir,
        start_frame=source["source_frames"][0], end_frame=source["source_frames"][1],
        complete_segment=True, marker_tip=np.asarray(source["T_marker0_usb_tip"]),
        sim_config=SimConfig(max_steps=1000, geometry=GEOMETRY),
        write_video=False, lowpass_hz=lowpass_hz)
    return replay_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lowpass-hz", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    dataset_id = f"usb_15x7_no_funnel_lowpass{args.lowpass_hz:g}hz_v1"
    output_dir = args.output_dir or Path("outputs/datasets") / dataset_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty dataset directory: {output_dir}")
    source_manifest = json.loads((SOURCE_DATASET / "capture_manifest.json").read_text())
    if len(source_manifest["episodes_manifest"]) != 72:
        raise ValueError("Expected the 72 reviewed v2 demonstrations")

    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=f"mimicvla/{dataset_id}", fps=20, features=FEATURES,
        root=output_dir, robot_type="franka_panda_usb",
        use_videos=True, video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=23, preset="veryfast", g=20))
    kept, rejected = [], []
    try:
        for episode in source_manifest["episodes_manifest"]:
            key = (episode["batch"], episode["clip"])
            replay_dir, summary = smoothed_replay(episode, args.lowpass_hz)
            replay = json.loads((replay_dir / "retargeted_replay.json").read_text())
            first = summary["initial_alignment_steps"]
            env = PandaUSBEnv(SimConfig(max_steps=1000, geometry=GEOMETRY,
                                        insertion_depth_m=.007))
            rows, failure = [], None
            first_7mm = first_12mm = None
            peak_force_n = 0.0
            try:
                for index, entry in enumerate(replay):
                    action = np.asarray(entry["action"], dtype=float)
                    if index >= first:
                        obs = env.observation(render=True)
                        rows.append({"observation.state": obs["observation.state"],
                                     "observation.images.camera1":
                                         obs["observation.images.camera1"],
                                     "observation.images.camera2":
                                         obs["observation.images.camera2"],
                                     "action": action.astype(np.float32), "task": TASK})
                    _, _, info = env.step(action)
                    peak_force_n = max(peak_force_n, info["max_wall_normal_force_n"])
                    joints = env.observation(render=False)["observation.state"][:7]
                    limits = env.model.jnt_range[:7]
                    if info["bad_collision"]:
                        failure = f"deep collision at step {index}"
                        break
                    if (np.any(joints < limits[:, 0] - 1e-4) or
                            np.any(joints > limits[:, 1] + 1e-4)):
                        failure = f"joint limit at step {index}"
                        break
                    if index >= first and info["insertion_success"] and first_7mm is None:
                        first_7mm = index
                    if (index >= first and first_12mm is None and
                            insertion_success_from_metrics(
                                info, required_depth_m=.012, geometry=GEOMETRY)):
                        first_12mm = index
                if failure is None and info["axial_error_m"] < .012:
                    failure = f"final depth {1000 * info['axial_error_m']:.1f} mm"
            finally:
                env.close()
            if failure is not None:
                rejected.append({"batch": key[0], "clip": key[1], "reason": failure})
                print(episode["clip"], "rejected:", failure, flush=True)
                continue
            for row in rows:
                dataset.add_frame(row)
            dataset.save_episode()
            kept.append({
                "episode_index": len(kept), "batch": key[0], "clip": key[1],
                "split": "holdout" if key in HOLDOUT_CLIPS else "train",
                "v2_episode_index": episode["episode_index"],
                "source_track": episode["source_track"], "replay_dir": str(replay_dir),
                "calibration_epoch": episode["calibration_epoch"],
                "human_review_decision": "accept", "frames": len(rows),
                "first_source_step": first, "last_source_step": len(replay) - 1,
                "first_7mm_step": first_7mm, "first_12mm_step": first_12mm,
                "final_depth_m": info["axial_error_m"],
                "peak_wall_force_n_full_replay": peak_force_n,
            })
            print(episode["clip"], len(rows), "frames", flush=True)
    finally:
        dataset.finalize()

    manifest = {"selection": "the 72 reviewed v2 demonstrations, re-retargeted with a "
                             "low-pass on the tracked tip pose; v2 gate re-applied",
                "lowpass": {"type": "zero-phase 2nd-order Butterworth (filtfilt)",
                            "cutoff_hz": args.lowpass_hz,
                            "applied_to": "socket-relative tip position and orientation, "
                                          "resampled to 20 Hz, before retargeting"},
                "episodes": len(kept), "frames": sum(row["frames"] for row in kept),
                "holdout_episodes": [row["episode_index"] for row in kept
                                     if row["split"] == "holdout"],
                "geometry": vars(GEOMETRY), "camera_setup": CAMERA_SETUP,
                "debug_sites_visible": False, "source_dataset": str(SOURCE_DATASET),
                "gate": "no flagged deep collision or joint-limit failure over the whole "
                        "replay, and finishes at least 12 mm deep",
                "rejected": rejected, "episodes_manifest": kept}
    (output_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"episodes": manifest["episodes"], "frames": manifest["frames"],
                      "rejected": len(rejected)}))


if __name__ == "__main__":
    main()
