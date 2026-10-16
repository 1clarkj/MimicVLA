"""Turn tracked human recordings into a LeRobot dataset of Panda demonstrations.

    mimicvla build-dataset TRACKING [--clips LIST] --output outputs/datasets/NAME

TRACKING is a folder of ``mimicvla track`` outputs (``<clip>/trajectory.jsonl``)
or of ``<clip>.jsonl`` files, as in the published raw dataset. Each recording:

1. selects its longest continuous insertion segment (at least 20 tracked
   frames, starting outside the port and gaining at least 10 mm of depth);
2. replays that plug-tip trajectory on the Panda in MuJoCo, after a short
   alignment from the reset pose to the first tracked pose;
3. keeps the episode only if the tip centre passes 7 mm deep during the human
   motion, finishes at least 12 mm deep, and the replay never penetrates the
   port walls by more than 1 mm or reaches a joint limit;
4. re-executes the same actions with the two policy cameras rendering and
   saves state, images and actions from the end of the alignment onward.

``--clips`` is a text file with one clip name per line: only those clips are
used, in that order (for example the clips accepted in a manual review).
Replays and their videos go to ``<output>_replays/<clip>/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from mimicvla.dataset.lerobot import CAMERA_SETUP, FEATURES, TASK
from mimicvla.retarget import pose_motion_rates, replay_recording
from mimicvla.sim.env import PandaUSBEnv, SimConfig, insertion_success_from_metrics
from mimicvla.sim.scene import SocketGeometry

GATE_DEPTH_M = .007
FINAL_DEPTH_M = .012


def tracking_files(root: Path) -> dict[str, Path]:
    """Clip name -> trajectory file, for both track outputs and flat .jsonl folders."""
    files = {path.parent.name: path for path in root.glob("*/trajectory.jsonl")}
    files.update({path.stem: path for path in root.glob("*.jsonl")})
    if not files:
        raise ValueError(f"No tracking files under {root}")
    return dict(sorted(files.items()))


def _runs(records: list[dict]) -> list[list[dict]]:
    """Continuous tracked runs; a pose jump above 0.75 m/s or 8 rad/s splits a run."""
    runs, current = [], []
    for record in records:
        if record["quality"] == "ok":
            if current:
                linear, angular = pose_motion_rates(
                    np.asarray(current[-1]["T_port_usb_marker"]),
                    np.asarray(record["T_port_usb_marker"]),
                    record["timestamp_s"] - current[-1]["timestamp_s"])
                if linear > .75 or angular > 8.0:
                    runs.append(current)
                    current = []
            current.append(record)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def select_insertion_run(records: list[dict], min_frames: int = 20
                         ) -> tuple[list[dict] | None, dict]:
    """The continuous run with the most insertion progress, or None."""
    candidates = []
    for run in _runs(records):
        if len(run) < min_frames:
            continue
        depths = median_filter(np.array([row["T_socket_usb_tip"][2][3] for row in run]),
                               size=3, mode="nearest")
        start_depth = float(np.median(depths[:3]))
        peak_depth = float(np.max(depths))
        progress = peak_depth - start_depth
        # A long loss of tracking can hide the approach. Keep the continuous
        # insertion after it when that run still shows substantial progress.
        if start_depth <= .005 and peak_depth >= .006 and progress >= .010:
            candidates.append((progress, len(run), run, {
                "start_depth_m": start_depth, "peak_depth_m": peak_depth,
                "axial_progress_m": progress}))
    if not candidates:
        return None, {"reason": "no continuous 20-frame run with 10 mm progress to 6 mm depth"}
    _, _, run, details = max(candidates, key=lambda item: (item[0], item[1]))
    return run, details


def replay_clip(track: Path, output_dir: Path, geometry: SocketGeometry, *,
                lowpass_hz: float | None = None, mirror: bool = False,
                video: bool = True) -> dict:
    """Replay one recording and apply the gate; returns a manifest row."""
    records = [json.loads(line) for line in track.read_text().splitlines()]
    if not any(record.get("T_socket_usb_tip") for record in records):
        raise ValueError(f"{track} has no T_socket_usb_tip; track it with `mimicvla track`")
    segment, details = select_insertion_run(records)
    row = {"source_track": str(track), **details}
    if segment is None:
        return {**row, "kept": False}
    summary = replay_recording(track, output_dir, start_frame=segment[0]["frame_index"],
                               end_frame=segment[-1]["frame_index"], complete_segment=True,
                               sim_config=SimConfig(max_steps=1000, geometry=geometry,
                                                    insertion_depth_m=GATE_DEPTH_M),
                               write_video=video, lowpass_hz=lowpass_hz,
                               mirror_socket_x=mirror)
    replay = json.loads((output_dir / "retargeted_replay.json").read_text())
    first = summary["initial_alignment_steps"]
    source = replay[first:]
    first_7mm = next((first + i for i, step in enumerate(source)
                      if step["metrics"]["insertion_success"]), None)
    first_12mm = next((first + i for i, step in enumerate(source)
                       if insertion_success_from_metrics(step["metrics"], geometry=geometry,
                                                         required_depth_m=FINAL_DEPTH_M)), None)
    final_depth = replay[-1]["metrics"]["axial_error_m"]
    failures = [reason for reason, failed in [
        ("no 7 mm insertion during the human motion", first_7mm is None),
        ("deep collision", summary["bad_collision_steps"] > 0),
        ("joint limit", not summary["all_joint_limits_ok"]),
        ("finishes shallower than 12 mm", final_depth < FINAL_DEPTH_M)] if failed]
    return {**row, "kept": not failures, "rejected_because": failures,
            "replay_dir": str(output_dir),
            "source_start_frame": segment[0]["frame_index"],
            "source_end_frame": segment[-1]["frame_index"],
            "first_source_step": first, "last_source_step": len(replay) - 1,
            "first_7mm_step": first_7mm, "first_12mm_step": first_12mm,
            "final_depth_m": final_depth,
            "peak_wall_force_n_full_replay": summary["max_wall_normal_force_n"]}


def capture(dataset, replay_dir: Path, geometry: SocketGeometry, first: int) -> int:
    """Re-execute a replay's actions with the policy cameras; returns saved frames."""
    replay = json.loads((replay_dir / "retargeted_replay.json").read_text())
    env = PandaUSBEnv(SimConfig(max_steps=1000, geometry=geometry,
                                insertion_depth_m=GATE_DEPTH_M))
    try:
        for name in ("ee_site", "usb_tip", "socket_target"):
            if env.model.site(name).rgba[3] != 0:
                raise ValueError(f"Debug site {name} would be visible in the images")
        for index, entry in enumerate(replay):
            action = np.asarray(entry["action"], dtype=float)
            if index >= first:
                obs = env.observation(render=True)
                dataset.add_frame({"observation.state": obs["observation.state"],
                                   "observation.images.camera1": obs["observation.images.camera1"],
                                   "observation.images.camera2": obs["observation.images.camera2"],
                                   "action": action.astype(np.float32), "task": TASK})
            env.step(action)
    finally:
        env.close()
    dataset.save_episode()
    return len(replay) - first


def build(tracking: Path, output: Path, *, clips: list[str] | None = None,
          port_mm: tuple[float, float] = (15, 7), lowpass_hz: float | None = None,
          mirror: bool = False, videos: bool = True) -> dict:
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty dataset directory: {output}")
    files = tracking_files(tracking)
    if clips is not None:
        missing = [clip for clip in clips if clip not in files]
        if missing:
            raise ValueError(f"No tracking for {missing[:5]}")
        files = {clip: files[clip] for clip in clips}
    geometry = SocketGeometry(inner_half_x_m=port_mm[0] / 2000, inner_half_y_m=port_mm[1] / 2000)
    replays = output.parent / f"{output.name}_replays"
    rows = []
    for clip, track in files.items():
        for mirrored in ([False, True] if mirror else [False]):
            name = f"{clip}_mirror" if mirrored else clip
            try:
                row = replay_clip(track, replays / name, geometry, lowpass_hz=lowpass_hz,
                                  mirror=mirrored, video=videos)
            except ValueError as error:
                row = {"source_track": str(track), "kept": False,
                       "rejected_because": [str(error)]}
            rows.append({"clip": name, **row})
            print(name, "kept" if row["kept"] else
                  f"rejected: {row.get('rejected_because') or row.get('reason')}", flush=True)

    dataset = LeRobotDataset.create(
        repo_id=f"local/{output.name}", fps=20, features=FEATURES, root=output,
        robot_type="franka_panda_usb", use_videos=True, video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=23, preset="veryfast", g=20))
    episodes = []
    try:
        for row in rows:
            if not row["kept"]:
                continue
            frames = capture(dataset, Path(row["replay_dir"]), geometry, row["first_source_step"])
            episodes.append({"episode_index": len(episodes), "frames": frames,
                             **{key: value for key, value in row.items()
                                if key not in ("kept", "rejected_because")}})
    finally:
        dataset.finalize()
    manifest = {"episodes": len(episodes), "frames": sum(e["frames"] for e in episodes),
                "geometry": {"inner_half_x_m": geometry.inner_half_x_m,
                             "inner_half_y_m": geometry.inner_half_y_m},
                "lowpass_hz": lowpass_hz, "mirror": mirror,
                "gate": "tip centre passes 7 mm during the human motion; no deep collision "
                        "or joint limit; finishes at least 12 mm deep",
                "camera_setup": CAMERA_SETUP, "episodes_manifest": episodes,
                "rejected": [{"clip": row["clip"],
                              "because": row.get("rejected_because") or [row.get("reason")]}
                             for row in rows if not row["kept"]]}
    (output / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output": str(output), "episodes": manifest["episodes"],
            "frames": manifest["frames"], "rejected": len(manifest["rejected"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    run(parser.parse_args())


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tracking", type=Path, help="folder of tracked recordings")
    parser.add_argument("--output", type=Path, required=True, help="new LeRobot dataset folder")
    parser.add_argument("--clips", type=Path, help="text file: clips to use, in order")
    parser.add_argument("--port-mm", type=float, nargs=2, default=(15, 7),
                        metavar=("WIDTH", "HEIGHT"), help="simulated port opening (default 15 7)")
    parser.add_argument("--lowpass-hz", type=float,
                        help="low-pass the tracked poses at this cutoff before retargeting")
    parser.add_argument("--mirror", action="store_true",
                        help="also add each clip mirrored across the port's vertical mid-plane")
    parser.add_argument("--no-videos", action="store_true", help="skip the replay videos")


def run(args: argparse.Namespace) -> None:
    clips = ([line.strip() for line in args.clips.read_text().splitlines() if line.strip()]
             if args.clips else None)
    print(json.dumps(build(args.tracking, args.output, clips=clips, port_mm=tuple(args.port_mm),
                           lowpass_hz=args.lowpass_hz, mirror=args.mirror,
                           videos=not args.no_videos), indent=2))


if __name__ == "__main__":
    main()
