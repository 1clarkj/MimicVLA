"""Export reviewed, clean guided-replay prefixes as a local LeRobot dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mimicvla.dataset.lerobot import TASK
from mimicvla.sim.env import PandaUSBEnv, SimConfig
from mimicvla.sim.scene import SocketGeometry, TRAINING_SOCKET


WIDE_NO_FUNNEL = SocketGeometry(inner_half_x_m=.009, inner_half_y_m=.005,
                                mouth_half_x_m=.009, mouth_half_y_m=.005)
CAMERA_SETUP = {
    "observation.images.camera1": {"mujoco_camera": "front_rgb",
                                   "world_position_m": [0.52, -0.28, 0.72],
                                   "looks_at_socket_center": True, "vertical_fov_deg": 30},
    "observation.images.camera2": {"mujoco_camera": "wrist_rgb",
                                   "hand_position_m": [0, -0.045, 0.070],
                                   "hand_target_m": [0, 0, 0.160], "vertical_fov_deg": 70},
}


FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (9,), "names": [
        *(f"joint{i}_rad" for i in range(1, 8)), "finger1_m", "finger2_m"]},
    "action": {"dtype": "float32", "shape": (6,), "names": [
        "dx_world_m", "dy_world_m", "dz_world_m",
        "rx_world_rad", "ry_world_rad", "rz_world_rad"]},
    "observation.images.camera1": {"dtype": "video", "shape": (256, 256, 3),
                                   "names": ["height", "width", "channel"]},
    "observation.images.camera2": {"dtype": "video", "shape": (256, 256, 3),
                                   "names": ["height", "width", "channel"]},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", choices=("reviewed", "technical"), default="reviewed")
    parser.add_argument("--geometry", choices=("training", "wide_no_funnel"),
                        default="training")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-wall-force-n", type=float)
    args = parser.parse_args()
    if args.geometry == "wide_no_funnel" and args.selection != "reviewed":
        parser.error("wide_no_funnel export requires human-reviewed selection")
    force_gate_n = (100.0 if args.selection == "technical" and args.max_wall_force_n is None
                    else args.max_wall_force_n)
    if force_gate_n is not None and force_gate_n <= 0:
        parser.error("--max-wall-force-n must be positive")
    dataset_id = ("usb_wide_no_funnel_reviewed_all_v2" if args.geometry == "wide_no_funnel" else
                  "usb_guided_reviewed_v2" if args.selection == "reviewed" else
                  "usb_guided_technical_lowforce_v3")
    if args.output_dir is None:
        args.output_dir = Path("outputs/datasets") / dataset_id
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty dataset directory: {args.output_dir}")
    results_root = Path("outputs/recalibration")
    rows = json.loads((results_root / "replay_results.json").read_text())
    decisions_path = Path("outputs/trial_review/decisions.csv")
    with decisions_path.open(newline="") as stream:
        decisions = {row["stem"]: row for row in csv.DictReader(stream)}
    chosen = [row for row in rows if row.get("training_prefix_candidate")
              and (args.selection == "technical" or
                   (decisions.get(row["clip"], {}).get("decision") == "accept"
                    and row["batch"] == "dual_original"))]
    if args.geometry == "wide_no_funnel":
        queue = {}
        for queue_name in ("wide_no_funnel_review_queue.csv",
                           "additional_wide_no_funnel_review_queue.csv"):
            with (results_root / queue_name).open(newline="") as stream:
                for item in csv.DictReader(stream):
                    key = (item.get("batch", "dual_original"), item["clip"])
                    if key in queue:
                        raise ValueError(f"Duplicate no-funnel review candidate: {key}")
                    queue[key] = item
        chosen = [row for row in rows
                  if (row["batch"], row["clip"]) in queue
                  and decisions.get(row["clip"], {}).get("decision") == "accept"]
        selected_keys = {(row["batch"], row["clip"]) for row in chosen}
        accepted_keys = {key for key in queue
                         if decisions.get(key[1], {}).get("decision") == "accept"}
        if selected_keys != accepted_keys or len(chosen) != len(selected_keys):
            raise ValueError("Accepted no-funnel review clips do not match unique replay rows")
    if args.selection == "technical":
        fallback_path = results_root / "fallback_results.json"
        if fallback_path.exists():
            chosen.extend(row for row in json.loads(fallback_path.read_text())
                          if row.get("training_prefix_candidate"))
    if force_gate_n is not None:
        if args.geometry == "wide_no_funnel":
            chosen = [row for row in chosen
                      if float(queue[(row["batch"], row["clip"])][
                          "peak_contact_n_to_first_entry"]) <= force_gate_n]
        else:
            contact_rows = json.loads((results_root / "guided_contact_audit.json").read_text())
            contact = {(row["batch"], row["clip"], row["tracking_mode"]): row
                       for row in contact_rows}
            chosen = [row for row in chosen
                      if contact[(row["batch"], row["clip"],
                                  "marker0_fallback" if row.get("tracking_mode") == "marker0_only"
                                  else "primary")]["max_wall_force_n_before_success"] <= force_gate_n]
    if not chosen:
        raise ValueError("No reviewed and collision-clean replay prefixes for this geometry")
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=f"mimicvla/{dataset_id}", fps=20,
        features=FEATURES, root=args.output_dir,
        robot_type="franka_panda_usb", use_videos=True, video_backend="pyav",
        rgb_encoder=(RGBEncoderConfig(vcodec="h264", crf=23, preset="veryfast", g=20)
                     if args.selection == "technical" or args.geometry == "wide_no_funnel"
                     else None))
    manifest = []
    try:
        for row in chosen:
            replay_dir = results_root / "replays" / row["batch"] / row["clip"]
            if row.get("tracking_mode") == "marker0_only":
                replay_dir /= "marker0_fallback"
            replay_dir /= args.geometry
            replay = json.loads((replay_dir / "retargeted_replay.json").read_text())
            summary = json.loads((replay_dir / "summary.json").read_text())
            if args.geometry == "wide_no_funnel":
                geometry = summary["socket_geometry"]
                if any(geometry.get(key) != value for key, value in
                       vars(WIDE_NO_FUNNEL).items() if key != "funnel_solref_timeconst_s"):
                    raise ValueError(f"Unexpected socket geometry: {row['clip']}")
                if (not summary["source_motion_insertion_reached"] or
                        summary["source_prefix_bad_collision_steps"] or
                        not summary["source_prefix_joint_limits_ok"]):
                    raise ValueError(f"Unclean no-funnel source prefix: {row['clip']}")
            first = summary["initial_alignment_steps"]
            last = first + summary["source_steps_to_first_success"]
            if first >= last or last >= len(replay):
                raise ValueError(f"No usable source prefix: {row['clip']}")
            env = PandaUSBEnv(SimConfig(max_steps=1000, geometry=(
                WIDE_NO_FUNNEL if args.geometry == "wide_no_funnel" else TRAINING_SOCKET)))
            frames = 0
            max_state_difference = 0.0
            episode_max_wall_force = max(entry["metrics"]["max_wall_normal_force_n"]
                                         for entry in replay[:last + 1])
            max_funnel_penetration = max(entry["metrics"].get("max_funnel_penetration_m", 0)
                                         for entry in replay[:last + 1])
            try:
                for index, entry in enumerate(replay[:last + 1]):
                    if index >= first:
                        obs = env.observation(render=True)
                        dataset.add_frame({"observation.state": obs["observation.state"],
                                           "observation.images.camera1": obs["observation.images.camera1"],
                                           "observation.images.camera2": obs["observation.images.camera2"],
                                           "action": np.asarray(entry["action"], dtype=np.float32),
                                           "task": TASK})
                        frames += 1
                    _, _, info = env.step(np.asarray(entry["action"]))
                    difference = np.max(np.abs(env.observation(render=False)["observation.state"] -
                                               np.asarray(entry["state"])))
                    max_state_difference = max(max_state_difference, float(difference))
                    if difference > 1e-3:
                        raise ValueError(f"Replay diverged at {row['clip']} step {index}: {difference}")
                    if index < first and info["bad_collision"]:
                        raise ValueError(f"Initial alignment contacted port: {row['clip']}")
                    if index >= first and info["bad_collision"]:
                        raise ValueError(f"Damaging contact in reviewed prefix: {row['clip']}")
                if not info["insertion_success"]:
                    raise ValueError(f"Reviewed prefix did not reach insertion: {row['clip']}")
                dataset.save_episode()
                manifest.append({"episode_index": len(manifest), "clip": row["clip"],
                                 "batch": row["batch"],
                                 "tracking_mode": row.get("tracking_mode", "dual_or_single_primary"),
                                 "source_track": row["source_track"],
                                 "source_start_frame": row["start_frame"],
                                 "source_end_frame": row["end_frame"],
                                 "calibration_epoch": [row["epoch_first"], row["epoch_last"]],
                                 "frames": frames,
                                 "max_state_replay_difference": max_state_difference,
                                 "max_wall_force_n_before_success": episode_max_wall_force,
                                 "max_funnel_penetration_mm_before_success":
                                     max_funnel_penetration * 1000,
                                 "human_review_decision": decisions.get(row["clip"], {}).get(
                                     "decision", "unreviewed"),
                                 "sim_geometry": args.geometry})
                print(row["clip"], frames, "frames", flush=True)
            finally:
                env.close()
    finally:
        dataset.finalize()
    result = {"selection": args.selection, "episodes": len(manifest),
              "frames": sum(item["frames"] for item in manifest),
              "max_wall_force_gate_n": force_gate_n,
              "geometry": vars(WIDE_NO_FUNNEL if args.geometry == "wide_no_funnel" else TRAINING_SOCKET),
              "camera_setup": CAMERA_SETUP,
              "gate": ("reviewed accept in either no-funnel queue; source motion reaches insertion with joint limits and no damaging contact through first success"
                       if args.geometry == "wide_no_funnel" else
                       "reviewed accept; source motion reaches insertion with joint limits and no damaging contact through first success"
                       if args.selection == "reviewed" else
                       f"automatic source-motion and collision gate; simulated wall force <= {force_gate_n:g} N; human review pending for most episodes"),
              "strict_evaluation": "separate; this dataset uses a widened training port",
              "episodes_manifest": manifest}
    (args.output_dir / "capture_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    review_note = ("Every source clip in this dataset was accepted in the local human review."
                   if args.selection == "reviewed" else
                   "Most source clips are technical candidates awaiting human tracking review. Use the reviewed dataset for a vetted baseline.")
    geometry_note = ("The training port has an 18 x 10 mm opening and no funnel. "
                     if args.geometry == "wide_no_funnel" else
                     "The training port is wider and has an 8 mm entry guide. ")
    (args.output_dir / "README.md").write_text(
        f"# Panda USB replay: {args.selection}, {args.geometry}\n\n"
        f"{review_note}\n\n"
        "The source iPhone videos supplied relative motion only. These episodes contain "
        "Panda state, Cartesian actions, and two simulated RGB cameras at 20 Hz. "
        "Each episode begins after simulator alignment to the first tracked pose and "
        "ends at the first training-port insertion with no damaging contact. "
        f"The optional simulated wall-force filter is {force_gate_n} N; this is a dataset heuristic, not a measured physical safety limit. "
        f"{geometry_note}"
        "Strict-port "
        "evaluation scores are stored separately in outputs/recalibration. "
        "See capture_manifest.json for clip, calibration epoch, tracking source, "
        "review decision, and exact geometry.\n")
    print(json.dumps({"episodes": result["episodes"], "frames": result["frames"]}))


if __name__ == "__main__":
    main()
