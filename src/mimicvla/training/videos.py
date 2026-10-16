"""Side-by-side README videos: iPhone + ArUco overlay | MuJoCo replay | policies.

For one start state, each panel runs at its own real time from the policy
hand-off state (the end of the replay's alignment prefix) and holds its last
frame, labelled with the outcome:

1. the source iPhone frames of the replayed segment, with the detected marker
   axes and the projected socket target and USB tip;
2. the human trajectory replayed by the Panda in MuJoCo;
3+. closed-loop SmolVLA rollouts from the same start (one seed each).

``--layout row`` puts every panel in one row; ``--layout grid`` puts the human
demo and its replay on top and the policies below. ``--sim-view`` picks what a
simulator panel shows (all extra viewpoints are free cameras; the policy's own
cameras are only rendered at a higher resolution):

- ``orbit``: close orbit view with the fixed policy camera as an inset (original);
- ``wide`` / ``close``: a fixed view of the whole arm and the port wall, or a
  three-quarter view of gripper and port, with front + wrist camera insets;
- ``cams``: the two policy cameras (front | wrist) side by side;
- ``wide+cams`` / ``close+cams``: that view above a front | wrist strip.

    MUJOCO_GL=egl python -m mimicvla.training.videos \
        --start all72:67 --policy raw=outputs/train/.../pretrained_model=raw_10k \
        --seed 1 --sim-view close+cams --gif-width 1000 \
        --output documentation/videos/holdout_single_marker_194743.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from mimicvla.training.evaluate import (GEOMETRY, SUCCESS_DEPTH_M, alignment_actions,
                                       load_policy, policy_action)
from mimicvla.sim.env import PandaUSBEnv, SimConfig
from mimicvla.vision.calibrate import load_calibration
from mimicvla.retarget import replay_recording
from mimicvla.vision.setup import load_setup
from mimicvla.vision.transforms import inverse


DATA = Path("data")
SETS = {"all72": DATA / "mimicvla-panda-usb",
        "excluded55": DATA / "mimicvla-panda-usb/eval_starts/excluded55"}
RAW = DATA / "mimicvla-human-usb-raw"
FPS = 20
SIM_PX = 480
HUMAN_W = 854
INSET_PX = 170
GREEN, RED, WHITE = (90, 220, 90), (235, 80, 80), (255, 255, 255)


# Free viewpoints: (lookat, azimuth, elevation, distance). The port wall is the
# plane x = 0.65 m facing -x; the robot base is at the origin.
VIEWS = {"orbit": ([.56, -.04, .6], 240, -22, .8),
         "wide": ([.37, 0, .42], 290, -12, 1.1),
         "close": ([.61, 0, .6], 300, -15, .32)}
SIM_VIEWS = ("orbit", "wide", "close", "cams", "wide+cams", "close+cams")
PANEL_W = 720          # panel width for every --sim-view except orbit
CAM_PX = PANEL_W // 2  # each policy camera in a front | wrist strip
BAND = False           # labels in a band above the frame instead of over it


def finish(frames: list[np.ndarray], title: str, status: tuple[str, tuple],
           running: tuple[str, tuple] = ("", WHITE)) -> list[np.ndarray]:
    """Title every frame; reveal the outcome only on the final (held) frame."""
    return ([label(f, [(title, WHITE), running]) for f in frames[:-1]] +
            [label(frames[-1], [(title, WHITE), status])])


def label(frame: np.ndarray, lines: list[tuple[str, tuple]]) -> np.ndarray:
    frame = frame.copy()
    if BAND:
        band = np.full((10 + 30 * len(lines), frame.shape[1], 3), 20, dtype=np.uint8)
        for row, (text, colour) in enumerate(lines):
            cv2.putText(band, text, (12, 30 + 30 * row), cv2.FONT_HERSHEY_SIMPLEX, .75,
                        colour, 2, cv2.LINE_AA)
        return np.concatenate([band, frame], axis=0)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 14 + 30 * len(lines)), (20, 20, 20), -1)
    for row, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (12, 34 + 30 * row), cv2.FONT_HERSHEY_SIMPLEX, .75,
                    colour, 2, cv2.LINE_AA)
    return frame


def tag(frame: np.ndarray, text: str, x: int, y: int, scale: float = .5) -> None:
    """Small white caption with a dark outline, readable on any background."""
    for colour, width in [((20, 20, 20), 4), (WHITE, 1)]:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, width,
                    cv2.LINE_AA)


def panel_size(view: str = "orbit") -> tuple[int, int]:
    """(width, height) of a simulator panel for a --sim-view."""
    if view == "orbit":
        return SIM_PX, SIM_PX
    if view == "cams":
        return PANEL_W, CAM_PX
    if view.endswith("+cams"):
        return PANEL_W, PANEL_W * 9 // 16 + CAM_PX
    return PANEL_W, PANEL_W * 2 // 3


class SimDrawer:
    """Renders one --sim-view of the current sim state."""

    def __init__(self, env: PandaUSBEnv, view: str = "orbit"):
        self.env, self.view, self.renderers = env, view, {}
        width, height = panel_size(view)
        env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, width, SIM_PX)
        env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, height, SIM_PX)

    def render(self, camera, width: int, height: int) -> np.ndarray:
        if (width, height) not in self.renderers:
            self.renderers[width, height] = mujoco.Renderer(self.env.model, height=height,
                                                            width=width)
        renderer = self.renderers[width, height]
        if isinstance(camera, str) and camera in VIEWS:
            lookat, azimuth, elevation, distance = VIEWS[camera]
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = lookat
            camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
        renderer.update_scene(self.env.data, camera=camera)
        return renderer.render().copy()

    def cams(self) -> np.ndarray:
        """Front | wrist policy cameras, as a PANEL_W x CAM_PX strip."""
        strip = np.concatenate([self.render("front_rgb", CAM_PX, CAM_PX),
                                self.render("wrist_rgb", CAM_PX, CAM_PX)], axis=1)
        strip[:, CAM_PX - 1:CAM_PX + 1] = 255
        tag(strip, "policy camera: front", 10, CAM_PX - 12, .7)
        tag(strip, "policy camera: wrist", CAM_PX + 10, CAM_PX - 12, .7)
        return strip

    def __call__(self) -> np.ndarray:
        view = self.view
        if view == "orbit":
            return self.legacy()
        if view == "cams":
            return self.cams()
        width, height = panel_size(view)
        if view.endswith("+cams"):
            top = self.render(view.removesuffix("+cams"), width, height - CAM_PX)
            return np.concatenate([top, self.cams()], axis=0)
        frame = self.render(view, width, height)
        inset = int(height * .36)
        # Both insets bottom-left: the wide view has empty floor there.
        for x, camera, text in [(8, "front_rgb", "front cam"),
                                (inset + 16, "wrist_rgb", "wrist cam")]:
            frame[height - inset - 8:height - 8, x:x + inset] = self.render(camera, inset, inset)
            cv2.rectangle(frame, (x - 1, height - inset - 9), (x + inset, height - 8), WHITE, 2)
            tag(frame, text, x + 6, height - 16, .6)
        return frame

    def legacy(self) -> np.ndarray:
        """Wide orbit view of the arm, with the policy's fixed camera as an inset."""
        frame = self.render("orbit", SIM_PX, SIM_PX)
        inset = cv2.resize(self.render("front_rgb", SIM_PX, SIM_PX), (INSET_PX, INSET_PX),
                           interpolation=cv2.INTER_AREA)
        frame[-INSET_PX - 8:-8, -INSET_PX - 8:-8] = inset
        cv2.rectangle(frame, (SIM_PX - INSET_PX - 9, SIM_PX - INSET_PX - 9),
                      (SIM_PX - 8, SIM_PX - 8), WHITE, 2)
        cv2.putText(frame, "policy camera", (SIM_PX - INSET_PX - 8, SIM_PX - INSET_PX - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, WHITE, 1, cv2.LINE_AA)
        return frame

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()


def sim_renderer(env: PandaUSBEnv, view: str = "orbit") -> SimDrawer:
    return SimDrawer(env, view)


def draw(renderer: SimDrawer, env: PandaUSBEnv) -> np.ndarray:
    return renderer()


def clip_source(clip: str) -> dict:
    """The clip's row in the raw dataset's metadata (video, setup, replayed frames)."""
    for line in (RAW / "videos/metadata.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["clip_id"] == clip:
            return row
    raise ValueError(f"{clip} is not in {RAW}")


def human_replay(clip: str, folder: Path) -> tuple[list[dict], int]:
    """Replay the clip's tracked segment; returns the steps and the alignment length."""
    source = clip_source(clip)
    summary = replay_recording(RAW / "tracking" / f"{clip}.jsonl", folder,
                               start_frame=source["replay_start_frame"],
                               end_frame=source["replay_end_frame"], complete_segment=True,
                               sim_config=SimConfig(max_steps=1000, geometry=GEOMETRY),
                               write_video=False)
    return (json.loads((folder / "retargeted_replay.json").read_text()),
            summary["initial_alignment_steps"])


def human_panel(clip: str, size: tuple[int, int] = (HUMAN_W, SIM_PX)) -> list[np.ndarray]:
    source = clip_source(clip)
    video = RAW / "videos" / source["file_name"].removeprefix("videos/")
    records = [json.loads(line) for line in
               (RAW / "tracking" / f"{clip}.jsonl").read_text().splitlines()]
    first, last = source["replay_start_frame"], source["replay_end_frame"]
    setup = load_setup(RAW / source["setup"])
    calibration = load_calibration(setup.camera)
    camera = np.asarray(calibration["camera_matrix"], dtype=float)
    distortion = np.asarray(calibration["distortion_coefficients"], dtype=float)
    port_mount = setup.T_socket_port_marker
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
        port = markers.get(str(setup.port_marker.id))
        if port is not None and record.get("T_socket_usb_tip") is not None:
            camera_socket = np.asarray(port["T_camera_marker"]) @ inverse(port_mount)
            for transform, colour, text in [
                    (camera_socket, (0, 255, 0), "socket"),
                    (camera_socket @ np.asarray(record["T_socket_usb_tip"]),
                     (255, 0, 255), "USB tip")]:
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
    # Crop a window of the panel's aspect (half the frame width, or the full
    # height if that is shorter) centred on the median socket.
    height, width = raw[0].shape[:2]
    crop_w = width // 2
    crop_h = min(height, crop_w * size[1] // size[0])
    crop_w = crop_h * size[0] // size[1]
    cx, cy = np.median(socket_px, axis=0).astype(int) if socket_px else (width // 2, height // 2)
    x0 = int(np.clip(cx - crop_w // 2, 0, width - crop_w))
    y0 = int(np.clip(cy - crop_h // 2, 0, height - crop_h))
    raw = [cv2.cvtColor(cv2.resize(f[y0:y0 + crop_h, x0:x0 + crop_w], size,
                                   interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
           for f in raw]
    # Resample source frames (24-30 fps) to the 20 Hz output clock.
    times = np.arange(int(len(raw) / fps * FPS))
    frames = [raw[min(len(raw) - 1, int(round(t * fps / FPS)))] for t in times]
    title = f"Human demo, iPhone + ArUco | {clip}"
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
              (f"failed: {1000 * np.hypot(info['socket_lateral_x_m'], info['socket_lateral_y_m']):.0f}"
               " mm off the port", RED))
    if BAND:
        status = (status[0] + (f" ({record.split(' seeds')[0]} eval seeds succeed)"
                               if record else ""),
                  status[1])
        return finish(frames, f"SmolVLA policy: {name}", status, (record, (180, 180, 180)))
    frames = finish(frames, f"SmolVLA policy: {name}", status)
    if record:
        for frame in frames:
            cv2.putText(frame, record, (12, SIM_PX - INSET_PX - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(frame, record, (12, SIM_PX - INSET_PX - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, WHITE, 1, cv2.LINE_AA)
    return frames


def compose(parts: list[np.ndarray], layout: str) -> np.ndarray:
    """One output frame: a single row, or human | replay over the policies."""
    rows = [parts] if layout == "row" else [parts[:2], parts[2:]]
    rows = [row for row in rows if row]
    width = max(sum(p.shape[1] for p in row) + 4 * (len(row) - 1) for row in rows)
    out = []
    for row in rows:
        height = max(p.shape[0] for p in row)
        cells = []
        for part in row:
            part = np.pad(part, ((0, height - part.shape[0]), (0, 0), (0, 0)))
            cells += [part, np.full((height, 4, 3), 255, dtype=np.uint8)]
        line = np.concatenate(cells[:-1], axis=1)
        pad = width - line.shape[1]
        out += [np.pad(line, ((0, 0), (pad // 2, pad - pad // 2), (0, 0))),
                np.full((4, width, 3), 255, dtype=np.uint8)]
    frame = np.concatenate(out[:-1], axis=0)
    return np.pad(frame, ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2), (0, 0)))


def write_gif(path: Path, frames: list[np.ndarray], duration_ms: float) -> None:
    """GIF with one shared palette and no dithering, so the held (static) panels
    stay pixel-identical between frames and compress away."""
    from PIL import Image

    sample = np.concatenate(frames[::max(1, len(frames) // 8)], axis=0)
    colours = Image.fromarray(sample).quantize(colors=240, method=Image.Quantize.MEDIANCUT)
    # Reserve exact entries for the overlay / status colours (small areas that
    # median cut would otherwise merge into the background).
    fixed = [GREEN, RED, WHITE, (180, 180, 180), (20, 20, 20), (0, 255, 0), (255, 0, 255),
             (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)]
    entries = colours.getpalette()[:3 * 240] + [c for colour in fixed for c in colour]
    palette = Image.new("P", (1, 1))
    palette.putpalette(entries + [0] * (768 - len(entries)))
    images = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE)
              for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms,
                   loop=0, optimize=True, disposal=1)


def start_env(alignment: list) -> PandaUSBEnv:
    env = PandaUSBEnv(SimConfig(max_steps=10_000, geometry=GEOMETRY,
                                insertion_depth_m=SUCCESS_DEPTH_M))
    for action in alignment:
        env.step(np.asarray(action, dtype=float))
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
    parser.add_argument("--gif-fps", type=int, default=FPS // 2)
    parser.add_argument("--layout", choices=("row", "grid"), default="row",
                        help="row: all panels side by side; grid: human | replay on top, "
                             "policies below")
    parser.add_argument("--sim-view", choices=SIM_VIEWS, default="orbit")
    parser.add_argument("--stills", type=Path,
                        help="also write a start / mid / end contact sheet PNG here")
    args = parser.parse_args()
    global BAND
    BAND = args.layout == "grid" or args.sim_view != "orbit"
    sim_size = panel_size(args.sim_view)
    set_name, index = args.start.split(":")
    manifest = json.loads((SETS[set_name] / "capture_manifest.json").read_text())
    episode = manifest["episodes_manifest"][int(index)]
    with tempfile.TemporaryDirectory() as folder:
        replay, first = human_replay(episode["clip"], Path(folder))
    if first != episode["first_source_step"]:
        raise ValueError("The recomputed replay does not start where the evaluation starts")

    panels = [human_panel(episode["clip"], sim_size if BAND else (HUMAN_W, sim_size[1]))]
    env = start_env([entry["action"] for entry in replay[:first]])
    renderer = sim_renderer(env, args.sim_view)
    panels.append(replay_panel(env, renderer, replay, first))
    renderer.close(); env.close()
    for spec in args.policy:
        name, checkpoint, *eval_name = spec.split("=")
        record = (seed_record(eval_name[0], set_name, episode["episode_index"])
                  if eval_name else "")
        # The evaluated start state exactly (the recomputed replay can differ by ~1e-7).
        env = start_env(alignment_actions(episode))
        renderer = sim_renderer(env, args.sim_view)
        panels.append(policy_panel(env, renderer, name, Path(checkpoint), args.seed,
                                   args.max_steps, args.device, record))
        renderer.close(); env.close()

    length = max(len(p) for p in panels) + FPS  # hold the final frames for 1 s
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = [compose([panel[min(t, len(panel) - 1)] for panel in panels], args.layout)
              for t in range(length)]
    with imageio.get_writer(args.output, fps=FPS, codec="libx264", quality=7,
                            macro_block_size=2) as writer:
        for frame in frames:
            writer.append_data(frame)
    scale = args.gif_width / frames[0].shape[1]
    size = (args.gif_width, int(round(frames[0].shape[0] * scale / 2) * 2))
    stride = max(1, round(FPS / args.gif_fps))
    write_gif(args.output.with_suffix(".gif"),
              [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames[::stride]],
              1000 * stride / FPS)
    if args.stills:
        args.stills.parent.mkdir(parents=True, exist_ok=True)
        picks = [0, int(.8 * min(len(p) for p in panels[1:])), length - 1]
        sheet = [cv2.resize(frames[t], (frames[t].shape[1] // 2, frames[t].shape[0] // 2),
                            interpolation=cv2.INTER_AREA) for t in picks]
        gap = np.full((sheet[0].shape[0], 6, 3), 255, dtype=np.uint8)
        imageio.imwrite(args.stills, np.concatenate([sheet[0], gap, sheet[1], gap, sheet[2]],
                                                    axis=1))
    print(json.dumps({"output": str(args.output), "frames": length,
                      "panels": [len(p) for p in panels]}))


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(Path("outputs/hf_cache").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
