"""Side-by-side README videos: iPhone + ArUco overlay | MuJoCo replay | policies.

For one start state, each panel runs at its own real time from the policy
hand-off state (the end of the replay's alignment prefix) and holds its last
frame, labelled with the outcome:

1. the source iPhone frames of the replayed segment, with the detected marker
   axes and the projected socket target and USB tip;
2. the human trajectory replayed by the Panda in MuJoCo;
3+. closed-loop SmolVLA rollouts from the same start (one seed each).

    MUJOCO_GL=egl python -m mimicvla.training.videos \
        --start all72:67 --policy lowpass_20k=outputs/train/.../pretrained_model \
        --output documentation/videos/holdout_67.mp4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from mimicvla.training.evaluate import GEOMETRY, SUCCESS_DEPTH_M, load_policy, policy_action
from mimicvla.sim.env import PandaUSBEnv, SimConfig
from mimicvla.vision.calibrate import load_calibration
from mimicvla.vision.target_frames import provisional_mounts
from mimicvla.vision.transforms import inverse


SETS = {"all72": Path("outputs/datasets/usb_15x7_no_funnel_full_v2"),
        "excluded55": Path("outputs/eval_sets/excluded_starts_v1")}
VIDEO_DIRS = {"single_20mm": ("video_data/20mm_marker", "20mm_"),
              "dual_original": ("video_data/double_marker_original_mount", ""),
              "dual_minus3": ("video_data/double_marker", "")}
CALIBRATION = Path("outputs/iphone_calibration.json")
FPS = 20
SIM_PX = 480
HUMAN_W = 854
INSET_PX = 170
GREEN, RED, WHITE = (90, 220, 90), (235, 80, 80), (255, 255, 255)


def finish(frames: list[np.ndarray], title: str, status: tuple[str, tuple]) -> list[np.ndarray]:
    """Title every frame; reveal the outcome only on the final (held) frame."""
    return ([label(f, [(title, WHITE), ("", WHITE)]) for f in frames[:-1]] +
            [label(frames[-1], [(title, WHITE), status])])


def label(frame: np.ndarray, lines: list[tuple[str, tuple]]) -> np.ndarray:
    frame = frame.copy()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 14 + 30 * len(lines)), (20, 20, 20), -1)
    for row, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (12, 34 + 30 * row), cv2.FONT_HERSHEY_SIMPLEX, .75,
                    colour, 2, cv2.LINE_AA)
    return frame


def sim_renderer(env: PandaUSBEnv) -> mujoco.Renderer:
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, SIM_PX)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, SIM_PX)
    return mujoco.Renderer(env.model, height=SIM_PX, width=SIM_PX)


def draw(renderer: mujoco.Renderer, env: PandaUSBEnv) -> np.ndarray:
    """Wide orbit view of the arm, with the policy's fixed camera as an inset."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [.56, -.04, .6]
    camera.azimuth, camera.elevation, camera.distance = 240, -22, .8
    renderer.update_scene(env.data, camera=camera)
    frame = renderer.render().copy()
    renderer.update_scene(env.data, camera="front_rgb")
    inset = cv2.resize(renderer.render(), (INSET_PX, INSET_PX), interpolation=cv2.INTER_AREA)
    frame[-INSET_PX - 8:-8, -INSET_PX - 8:-8] = inset
    cv2.rectangle(frame, (SIM_PX - INSET_PX - 9, SIM_PX - INSET_PX - 9), (SIM_PX - 8, SIM_PX - 8),
                  WHITE, 2)
    cv2.putText(frame, "policy camera", (SIM_PX - INSET_PX - 8, SIM_PX - INSET_PX - 16),
                cv2.FONT_HERSHEY_SIMPLEX, .5, WHITE, 1, cv2.LINE_AA)
    return frame


def human_panel(episode: dict, replay_dir: Path) -> list[np.ndarray]:
    summary = json.loads((replay_dir / "summary.json").read_text())
    folder, prefix = VIDEO_DIRS[episode["batch"]]
    video = Path(folder) / f"{episode['clip'].removeprefix(prefix)}.mov"
    records = [json.loads(line) for line in Path(summary["source"]).read_text().splitlines()]
    first, last = summary["source_frames"]
    calibration = load_calibration(CALIBRATION)
    camera = np.asarray(calibration["camera_matrix"], dtype=float)
    distortion = np.asarray(calibration["distortion_coefficients"], dtype=float)
    port_mount, _ = provisional_mounts()
    marker_tip = np.asarray(summary["T_marker0_usb_tip"])
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    raw, socket_px = [], []
    for index in range(first, last + 1):
        ok, frame = capture.read()
        if not ok:
            break
        record = records[index]
        scale = frame.shape[1] / 1920
        markers = record.get("markers") or {}
        for key, marker in markers.items():
            pose = np.asarray(marker["T_camera_marker"])
            rvec, _ = cv2.Rodrigues(pose[:3, :3])
            cv2.drawFrameAxes(frame, camera, distortion, rvec, pose[:3, 3], .02,
                              max(2, int(3 * scale)))
        port = markers.get("port", markers.get("1"))
        if port is not None and record.get("T_port_usb_marker") is not None:
            camera_port = np.asarray(port["T_camera_marker"])
            camera_usb = camera_port @ np.asarray(record["T_port_usb_marker"])
            for transform, colour, text in [
                    (camera_port @ inverse(port_mount), (0, 255, 0), "socket"),
                    (camera_usb @ marker_tip, (255, 0, 255), "USB tip")]:
                rvec, _ = cv2.Rodrigues(transform[:3, :3])
                pixels, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, transform[:3, 3],
                                              camera, distortion)
                x, y = np.rint(pixels[0, 0]).astype(int)
                if text == "socket":
                    socket_px.append((x, y))
                cv2.circle(frame, (x, y), int(14 * scale), colour, max(2, int(4 * scale)))
                cv2.putText(frame, text, (x + int(20 * scale), y - int(20 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, colour,
                            max(2, int(3 * scale)), cv2.LINE_AA)
        raw.append(frame)
    capture.release()
    # Crop a 16:9 window (half the frame width) centred on the median socket.
    height, width = raw[0].shape[:2]
    crop_w, crop_h = width // 2, width * 9 // 32
    cx, cy = np.median(socket_px, axis=0).astype(int) if socket_px else (width // 2, height // 2)
    x0 = int(np.clip(cx - crop_w // 2, 0, width - crop_w))
    y0 = int(np.clip(cy - crop_h // 2, 0, height - crop_h))
    raw = [cv2.cvtColor(cv2.resize(f[y0:y0 + crop_h, x0:x0 + crop_w], (HUMAN_W, SIM_PX),
                                   interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
           for f in raw]
    # Resample source frames (24-30 fps) to the 20 Hz output clock.
    times = np.arange(int(len(raw) / fps * FPS))
    frames = [raw[min(len(raw) - 1, int(round(t * fps / FPS)))] for t in times]
    title = f"Human demo, iPhone + ArUco | {episode['clip']}"
    return [label(f, [(title, WHITE), (f"source frames {first}-{last}", WHITE)])
            for f in frames]


def replay_panel(env: PandaUSBEnv, renderer, replay: list[dict], first: int) -> list[np.ndarray]:
    frames, success = [], False
    for entry in replay[first:]:
        frames.append(draw(renderer, env))
        _, _, info = env.step(np.asarray(entry["action"], dtype=float))
        success = success or info["axial_error_m"] >= SUCCESS_DEPTH_M and info["insertion_success"]
    frames.append(draw(renderer, env))
    status = ("reaches 12 mm", GREEN) if success else ("does not insert in sim", RED)
    return finish(frames, "Panda replay of the human motion", status)


def seed_record(eval_name: str, set_name: str, episode_index: int) -> str:
    """'k/n seeds' for this start in the multi-seed evaluation, if available."""
    outcomes = []
    for path in sorted(Path("outputs/policy_eval_multiseed", eval_name).glob(
            f"{set_name}_seed*/summary.json")):
        outcomes += [row["success"] for row in json.loads(path.read_text())["results"]
                     if row["episode_index"] == episode_index]
    return f"{sum(outcomes)}/{len(outcomes)} seeds succeed in evaluation" if outcomes else ""


def policy_panel(env: PandaUSBEnv, renderer, name: str, checkpoint: Path, seed: int,
                 max_steps: int, device: str, record: str = "") -> list[np.ndarray]:
    import torch

    torch.manual_seed(seed)
    policy, pre, post = load_policy(checkpoint, device, 10)
    policy.reset()
    frames, result = [], None
    for step in range(max_steps):
        frames.append(draw(renderer, env))
        obs = env.observation(render=True)
        _, _, info = env.step(policy_action(policy, pre, post, obs, device))
        if info["insertion_success"]:
            result = step + 1
            break
    frames.append(draw(renderer, env))
    del policy
    status = ((f"inserted 12 mm in {result / FPS:.1f} s", GREEN) if result else
              (f"failed (lateral {1000 * info['socket_lateral_x_m']:+.0f}/"
               f"{1000 * info['socket_lateral_y_m']:+.0f} mm, "
               f"{info['max_wall_normal_force_n']:.0f} N)", RED))
    frames = finish(frames, f"SmolVLA policy: {name}", status)
    if record:
        for frame in frames:
            cv2.putText(frame, record, (12, SIM_PX - INSET_PX - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(frame, record, (12, SIM_PX - INSET_PX - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, WHITE, 1, cv2.LINE_AA)
    return frames


def start_env(replay: list[dict], first: int) -> PandaUSBEnv:
    env = PandaUSBEnv(SimConfig(max_steps=10_000, geometry=GEOMETRY,
                                insertion_depth_m=SUCCESS_DEPTH_M))
    for entry in replay[:first]:
        env.step(np.asarray(entry["action"], dtype=float))
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="SET:EPISODE, e.g. all72:67")
    parser.add_argument("--policy", action="append", default=[],
                        help="NAME=CHECKPOINT[=EVAL_NAME]; EVAL_NAME labels the panel with "
                             "this start's multi-seed evaluation record")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gif-width", type=int, default=1200)
    args = parser.parse_args()
    set_name, index = args.start.split(":")
    manifest = json.loads((SETS[set_name] / "capture_manifest.json").read_text())
    episode = manifest["episodes_manifest"][int(index)]
    replay_dir = Path(episode.get("replay_dir") or Path("outputs/recalibration/replays") /
                      episode["batch"] / episode["clip"] / "wide_no_funnel")
    replay = json.loads((replay_dir / "retargeted_replay.json").read_text())
    first = episode["first_source_step"]

    panels = [human_panel(episode, replay_dir)]
    env = start_env(replay, first)
    renderer = sim_renderer(env)
    panels.append(replay_panel(env, renderer, replay, first))
    renderer.close(); env.close()
    for spec in args.policy:
        name, checkpoint, *eval_name = spec.split("=")
        record = (seed_record(eval_name[0], set_name, episode["episode_index"])
                  if eval_name else "")
        env = start_env(replay, first)
        renderer = sim_renderer(env)
        panels.append(policy_panel(env, renderer, name, Path(checkpoint), args.seed,
                                   args.max_steps, args.device, record))
        renderer.close(); env.close()

    length = max(len(p) for p in panels) + FPS  # hold the final frames for 1 s
    args.output.parent.mkdir(parents=True, exist_ok=True)
    divider = np.full((SIM_PX, 4, 3), 255, dtype=np.uint8)
    frames = []
    for t in range(length):
        parts = []
        for panel in panels:
            parts += [panel[min(t, len(panel) - 1)], divider]
        frames.append(np.concatenate(parts[:-1], axis=1))
    with imageio.get_writer(args.output, fps=FPS, codec="libx264", quality=7,
                            macro_block_size=2) as writer:
        for frame in frames:
            writer.append_data(frame)
    scale = args.gif_width / frames[0].shape[1]
    size = (args.gif_width, int(round(frames[0].shape[0] * scale / 2) * 2))
    imageio.mimsave(args.output.with_suffix(".gif"),
                    [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames[::2]],
                    duration=2 / FPS, loop=0)
    print(json.dumps({"output": str(args.output), "frames": length,
                      "panels": [len(p) for p in panels]}))


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(Path("outputs/hf_cache").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
