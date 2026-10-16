"""The `mimicvla` command."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio


def run_sim_demo(output: Path) -> dict:
    """Align above the socket, descend, and save an RGB rollout."""
    from .sim import PandaUSBEnv
    from .sim.demo import scripted_actions

    env = PandaUSBEnv()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=10, codec="libx264", quality=7)
    metrics = []
    try:
        for index, action in enumerate(scripted_actions(env)):
            _, _, info = env.step(action)
            metrics.append(info)
            if index % 2 == 0:
                writer.append_data(env.render())
        result = {
            "video": str(output),
            "steps": len(metrics),
            "alignment_reached": any(m["alignment_success"] for m in metrics),
            "insertion_reached": any(m["insertion_success"] for m in metrics),
            "final": metrics[-1],
        }
        output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        writer.close()
        env.close()


def main() -> None:
    from .dataset import build

    parser = argparse.ArgumentParser(prog="mimicvla")
    commands = parser.add_subparsers(dest="command", required=True)

    calibration = commands.add_parser("calibrate", help="camera intrinsics from a ChArUco video")
    calibration.add_argument("input", type=Path, help="video file or photo folder")
    calibration.add_argument("--output", type=Path, default=Path("outputs/camera.json"))
    calibration.add_argument("--stride", type=int, default=15, help="use every Nth frame")
    calibration.add_argument("--square-mm", type=float, default=30.0)
    calibration.add_argument("--marker-mm", type=float, default=22.0)

    mount = commands.add_parser("fit-marker-mount",
                                help="fit the transform between two plug markers into a setup")
    mount.add_argument("videos", type=Path, nargs="+")
    mount.add_argument("--setup", type=Path, required=True)

    tracking = commands.add_parser("track", help="track the plug relative to the port in videos")
    tracking.add_argument("videos", type=Path, nargs="+")
    tracking.add_argument("--setup", type=Path, required=True, help="setup JSON for this rig")
    tracking.add_argument("--output-dir", type=Path, default=Path("outputs/tracking"),
                          help="one subfolder per video (default outputs/tracking)")
    tracking.add_argument("--no-video", action="store_true", help="skip annotated.mp4")

    replay = commands.add_parser("replay", help="replay one tracked recording on the Panda")
    replay.add_argument("trajectory", type=Path, help="trajectory.jsonl written by `track`")
    replay.add_argument("--output-dir", type=Path, help="default: <trajectory folder>/replay")
    replay.add_argument("--port-mm", type=float, nargs=2, default=(15, 7),
                        metavar=("WIDTH", "HEIGHT"))

    build.add_arguments(commands.add_parser(
        "build-dataset", help="tracked recordings -> gated Panda replays -> LeRobot dataset",
        description=build.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))

    evaluate = commands.add_parser("evaluate", help="run a policy closed loop from start states")
    evaluate.add_argument("checkpoint", help="pretrained_model folder")
    evaluate.add_argument("--starts", choices=["excluded", "holdout", "dataset"], default="excluded",
                          help="excluded: 55 starts from recordings never trained on (default); "
                               "holdout: the 8 held-out clips; dataset: all 72")
    evaluate.add_argument("--seeds", type=int, nargs="+", default=[0])
    evaluate.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    evaluate.add_argument("--video", action="store_true", help="save a video per rollout")

    download = commands.add_parser("download", help="fetch the published data into data/")
    download.add_argument("--raw", action="store_true", help="also the 134 iPhone recordings (6.8 GB)")
    download.add_argument("--policy", action="store_true", help="also the trained SmolVLA policy")

    sim = commands.add_parser("sim-demo", help="render a scripted insertion")
    sim.add_argument("--output", type=Path, default=Path("outputs/sim_demo.mp4"))
    commands.add_parser("sim-view", help="open the interactive MuJoCo viewer")
    smoke = commands.add_parser("vision-smoke", help="track a generated marker video")
    smoke.add_argument("--output-dir", type=Path, default=Path("outputs/vision_smoke"))

    args = parser.parse_args()
    if args.command == "calibrate":
        from .vision.calibrate import calibrate
        result = calibrate(args.input, args.output, square_m=args.square_mm / 1000,
                           marker_m=args.marker_mm / 1000, stride=args.stride)
        print(json.dumps({"output": str(args.output), "accepted_views": len(result["accepted_views"]),
                          "rms_reprojection_px": result["rms_reprojection_px"],
                          "board_center_span_fraction_xy": result["board_center_span_fraction_xy"],
                          "warnings": result["warnings"]}, indent=2))
    elif args.command == "fit-marker-mount":
        from .vision.setup import load_setup
        from .vision.track import fit_marker_mount
        result = fit_marker_mount(args.videos, load_setup(args.setup))
        data = json.loads(args.setup.read_text())
        data["usb_marker_2"]["T_usb_marker_usb_marker_2"] = result["T_usb_marker_usb_marker_2"]
        args.setup.write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items()
                          if key != "T_usb_marker_usb_marker_2"}, indent=2))
    elif args.command == "track":
        from .vision.setup import load_setup
        from .vision.track import track_video
        setup = load_setup(args.setup)
        for video in args.videos:
            print(json.dumps(track_video(video, setup, args.output_dir / video.stem,
                                         annotate=not args.no_video), indent=2), flush=True)
    elif args.command == "replay":
        from .sim.scene import SocketGeometry
        geometry = SocketGeometry(inner_half_x_m=args.port_mm[0] / 2000,
                                  inner_half_y_m=args.port_mm[1] / 2000)
        output = args.output_dir or args.trajectory.parent / "replay"
        print(json.dumps(build.replay_clip(args.trajectory, output, geometry), indent=2))
    elif args.command == "build-dataset":
        build.run(args)
    elif args.command == "evaluate":
        dataset = Path("data/mimicvla-panda-usb-15x7")
        if not dataset.exists():
            raise SystemExit("Run `mimicvla download` first")
        start_dir, split = {"excluded": (dataset / "eval_starts/excluded55", "all"),
                            "holdout": (dataset, "holdout"), "dataset": (dataset, "all")}[args.starts]
        for seed in args.seeds:
            subprocess.run([sys.executable, "-m", "mimicvla.training.evaluate", args.checkpoint,
                            "--dataset-dir", str(start_dir), "--split", split, "--seed", str(seed),
                            "--output-dir", str(args.output_dir / f"{args.starts}_seed{seed}"),
                            *([] if args.video else ["--no-video"])], check=True)
    elif args.command == "download":
        from .dataset.publish.hub import download
        print(json.dumps(download(raw=args.raw, policy=args.policy), indent=2))
    elif args.command == "sim-demo":
        result = run_sim_demo(args.output)
        print(json.dumps(result, indent=2))
        if not result["alignment_reached"] or not result["insertion_reached"]:
            raise SystemExit("The predefined path did not reach the socket")
    elif args.command == "sim-view":
        from .sim.viewer import run_viewer
        run_viewer()
    elif args.command == "vision-smoke":
        from .vision.synthetic import run_smoke
        print(json.dumps(run_smoke(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
