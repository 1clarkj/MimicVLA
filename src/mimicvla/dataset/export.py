"""Capture complete, reviewed 15 x 7 mm replays with marker-free RGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mimicvla.dataset.export_review import CAMERA_SETUP, FEATURES
from mimicvla.dataset.lerobot import TASK
from mimicvla.sim.env import PandaUSBEnv, SimConfig, insertion_success_from_metrics
from mimicvla.sim.scene import SocketGeometry


DATASET_ID = "usb_15x7_no_funnel_full_v2"
GEOMETRY = SocketGeometry(inner_half_x_m=.0075, inner_half_y_m=.0035)
DEPTH_M = .007
SWEEP_PATH = Path("outputs/recalibration/no_funnel_7mm_port_sweep.json")
SOURCE_MANIFEST = Path("outputs/datasets/usb_wide_no_funnel_reviewed_all_v2/"
                       "capture_manifest.json")
FULL_AUDIT_PATH = Path("outputs/recalibration/compact_15x7_full_replay_audit.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/datasets") / DATASET_ID)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty dataset directory: {args.output_dir}")

    sweep = json.loads(SWEEP_PATH.read_text())
    match = [row for row in sweep["sizes"]
             if row["width_mm"] == 15 and row["height_mm"] == 7]
    if len(match) != 1 or sweep["depth_mm"] != 7:
        raise ValueError("Missing the reviewed 15 x 7 mm, 7 mm-depth sweep")
    candidates = {(row["batch"], row["clip"]): row for row in match[0]["episodes"]}
    source = json.loads(SOURCE_MANIFEST.read_text())
    selected = [episode for episode in source["episodes_manifest"]
                if candidates[(episode["batch"], episode["clip"])]["valid"]]
    if (len(source["episodes_manifest"]) != 103 or len(candidates) != 103 or
            len(selected) != match[0]["valid_episodes"] or len(selected) != 75 or
            any(episode["human_review_decision"] != "accept" for episode in selected)):
        raise ValueError("Reviewed selection and 15 x 7 mm sweep do not agree")
    audit = json.loads(FULL_AUDIT_PATH.read_text())
    audited = {(row["batch"], row["clip"]): row for row in audit["episodes"]}
    if len(audited) != 75 or set(audited) != {
            (episode["batch"], episode["clip"]) for episode in selected}:
        raise ValueError("Full-replay audit does not match the 7 mm selection")
    selected = [episode for episode in selected
                if (audited[(episode["batch"], episode["clip"])][
                    "first_bad_collision_step"] is None and
                    audited[(episode["batch"], episode["clip"])][
                        "first_joint_limit_step"] is None and
                    audited[(episode["batch"], episode["clip"])][
                        "final_depth_mm"] >= 12)]
    if len(selected) != 72:
        raise ValueError(f"Expected 72 clean complete demonstrations, got {len(selected)}")

    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=f"mimicvla/{DATASET_ID}", fps=20, features=FEATURES,
        root=args.output_dir, robot_type="franka_panda_usb",
        use_videos=True, video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=23, preset="veryfast", g=20))
    result_episodes = []
    try:
        for episode in selected:
            key = (episode["batch"], episode["clip"])
            sweep_row = candidates[key]
            replay_dir = (Path("outputs/recalibration/replays") / episode["batch"] /
                          episode["clip"] / "wide_no_funnel")
            replay = json.loads((replay_dir / "retargeted_replay.json").read_text())
            summary = json.loads((replay_dir / "summary.json").read_text())
            first = summary["initial_alignment_steps"]
            last = len(replay) - 1
            audit_row = audited[key]
            if last != audit_row["last_source_step"]:
                raise ValueError(f"Source replay length changed: {key}")
            if first >= last or last >= len(replay):
                raise ValueError(f"Invalid full source replay: {key}")
            env = PandaUSBEnv(SimConfig(max_steps=1000, geometry=GEOMETRY,
                                        insertion_depth_m=DEPTH_M))
            frames = 0
            peak_force_n = 0.0
            first_7mm = None
            first_12mm = None
            try:
                for name in ("ee_site", "usb_tip", "socket_target"):
                    if env.model.site(name).rgba[3] != 0:
                        raise ValueError(f"Visible debug site {name}: {key}")
                for index, entry in enumerate(replay[:last + 1]):
                    if index >= first:
                        obs = env.observation(render=True)
                        dataset.add_frame({"observation.state": obs["observation.state"],
                                           "observation.images.camera1":
                                               obs["observation.images.camera1"],
                                           "observation.images.camera2":
                                               obs["observation.images.camera2"],
                                           "action": np.asarray(entry["action"], dtype=np.float32),
                                           "task": TASK})
                        frames += 1
                    _, _, info = env.step(np.asarray(entry["action"], dtype=float))
                    peak_force_n = max(peak_force_n, info["max_wall_normal_force_n"])
                    if info["bad_collision"]:
                        raise ValueError(f"Deep collision at {key} step {index}")
                    joints = env.observation(render=False)["observation.state"][:7]
                    limits = env.model.jnt_range[:7]
                    if (np.any(joints < limits[:, 0] - 1e-4) or
                            np.any(joints > limits[:, 1] + 1e-4)):
                        raise ValueError(f"Joint limit at {key} step {index}")
                    if index >= first and info["insertion_success"] and first_7mm is None:
                        first_7mm = index
                    if (index >= first and first_12mm is None and
                            insertion_success_from_metrics(
                                info, required_depth_m=.012, geometry=GEOMETRY)):
                        first_12mm = index
                if (first_7mm != sweep_row["first_success_step"] or
                        first_7mm != audit_row["first_7mm_step"] or
                        first_12mm != audit_row["first_12mm_step"] or
                        info["axial_error_m"] < .012):
                    raise ValueError(f"Full insertion replay changed: {key}")
                dataset.save_episode()
                result_episodes.append({
                    "episode_index": len(result_episodes), "batch": episode["batch"],
                    "clip": episode["clip"], "source_track": episode["source_track"],
                    "source_start_frame": episode["source_start_frame"],
                    "source_end_frame": episode["source_end_frame"],
                    "calibration_epoch": episode["calibration_epoch"],
                    "tracking_mode": episode["tracking_mode"],
                    "human_review_decision": "accept", "frames": frames,
                    "first_source_step": first, "last_source_step": last,
                    "first_7mm_step": first_7mm, "first_12mm_step": first_12mm,
                    "final_depth_m": info["axial_error_m"],
                    "peak_wall_force_n_full_replay": peak_force_n,
                })
                print(episode["clip"], frames, "frames", flush=True)
            finally:
                env.close()
    finally:
        dataset.finalize()

    manifest = {"selection": "reviewed 15 x 7 mm complete clean source replays; "
                             "7 mm used only for eligibility",
                "episodes": len(result_episodes),
                "frames": sum(row["frames"] for row in result_episodes),
                "geometry": vars(GEOMETRY), "selection_depth_m": DEPTH_M,
                "episode_end_rule": "end of complete source motion, after alignment",
                "camera_setup": CAMERA_SETUP, "debug_sites_visible": False,
                "source_manifest": str(SOURCE_MANIFEST), "source_sweep": str(SWEEP_PATH),
                "source_full_audit": str(FULL_AUDIT_PATH),
                "gate": "human accepted; tip center passes 7 mm during source motion; "
                        "entire source replay has no flagged deep collision or joint-limit "
                        "failure and finishes at least 12 mm deep",
                "episodes_manifest": result_episodes}
    (args.output_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Panda USB: reviewed 15 x 7 mm no-funnel training replays\n\n"
        "This local LeRobot v3 dataset has two marker-free simulated RGB cameras, "
        "9 measured Panda joint/finger states, 6D Cartesian actions, and the USB "
        "insertion task at 20 Hz. Episodes start after simulator alignment and "
        "continue through the entire source motion into the 15 x 7 mm opening. "
        "The 7 mm criterion only selects clips; every saved episode finishes "
        "at least 12 mm deep. "
        "The port has no funnel. All source clips were accepted in human review. "
        "The saved source action sequences were replayed under this exact geometry; "
        "the visual debug sites are invisible. See capture_manifest.json for "
        "provenance, per-episode force, and geometry. These are demonstrated "
        "replays, not closed-loop learned-policy successes.\n")
    print(json.dumps({"episodes": manifest["episodes"], "frames": manifest["frames"]}))


if __name__ == "__main__":
    main()
