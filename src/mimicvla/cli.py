"""Small command-line entry points for the current prototype stage."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio

from .sim import PandaUSBEnv
from .sim.demo import scripted_actions
from .sim.viewer import run_viewer


def run_sim_demo(output: Path) -> dict:
    """Align above the socket, descend, and save an RGB rollout."""
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
    parser = argparse.ArgumentParser(prog="mimicvla")
    commands = parser.add_subparsers(dest="command", required=True)
    sim = commands.add_parser("sim-demo", help="render the Panda's predefined USB insertion path")
    sim.add_argument("--output", type=Path, default=Path("outputs/sim_demo.mp4"))
    commands.add_parser("sim-view", help="open the native interactive MuJoCo viewer")
    calibration = commands.add_parser("calibrate", aliases=["calibrate-camera"],
                                      help="calibrate the phone camera from a ChArUco video or photos")
    calibration.add_argument("input", type=Path, help="video file or photo directory")
    calibration.add_argument("--output", type=Path, default=Path("outputs/iphone_calibration.json"))
    calibration.add_argument("--stride", type=int, default=15, help="use every Nth photo/frame")
    calibration.add_argument("--square-mm", type=float, default=30.0)
    calibration.add_argument("--marker-mm", type=float, default=22.0)
    tracking = commands.add_parser("track", aliases=["track-video"],
                                   help="track the USB and port markers in a video")
    tracking.add_argument("input", help="video file, or camera index such as 0")
    tracking.add_argument("--calibration", type=Path, default=Path("outputs/iphone_calibration.json"))
    tracking.add_argument("--output-dir", type=Path,
                          help="default: outputs/tracking/<video name>")
    tracking.add_argument("--usb-marker-mm", type=float, default=20.0,
                          help="outer black side of the USB marker: 12, 15, 18 or 20 mm on the sheet")
    tracking.add_argument("--port-marker-mm", type=float, default=50.0)
    tracking.add_argument("--max-frames", type=int, default=None)
    smoke = commands.add_parser("vision-smoke", help="generate and track synthetic marker video")
    smoke.add_argument("--output-dir", type=Path, default=Path("outputs/vision_smoke"))
    inspect = commands.add_parser("inspect-marker-frames", help="project provisional USB and socket target frames")
    inspect.add_argument("--trajectory", type=Path, required=True)
    inspect.add_argument("--calibration", type=Path, required=True)
    inspect.add_argument("--frame-index", type=int, default=90)
    inspect.add_argument("--output", type=Path, default=Path("outputs/target_frame_inspection.jpg"))
    inspect.add_argument("--tip-x-mm", type=float, default=0.0,
                         help="marker 0 to USB tip X offset; use -3 for the new dual-marker mount")
    replay = commands.add_parser("replay", aliases=["replay-tracking"],
                                 help="replay a tracked recording on the Panda in MuJoCo")
    replay.add_argument("trajectory", type=Path, help="trajectory.jsonl written by `track`")
    replay.add_argument("--output-dir", type=Path, help="default: <trajectory folder>/replay")
    replay.add_argument("--start-frame", type=int)
    replay.add_argument("--end-frame", type=int)
    replay.add_argument("--complete-segment", action="store_true",
                        help="continue through all recorded targets after first insertion success")
    replay.add_argument("--tip-x-mm", type=float, default=0.0,
                        help="marker 0 to USB tip X offset; use -3 for the new dual-marker mount")
    evaluate = commands.add_parser("evaluate-recordings", help="track and replay all insertion videos")
    evaluate.add_argument("--video-dir", type=Path, default=Path("video_data"))
    evaluate.add_argument("--calibration", type=Path, default=Path("outputs/iphone_calibration.json"))
    evaluate.add_argument("--output-dir", type=Path, default=Path("outputs/video_evaluation_12mm_full_replay"))
    evaluate.add_argument("--reuse-existing-replays", action="store_true",
                          help="reuse completed replay summaries when selected frame range matches")
    evaluate.add_argument("--retrack", action="store_true",
                          help="re-estimate marker poses even when cached tracking exists")
    dual = commands.add_parser("evaluate-dual-recordings",
                               help="compare ID 0 only and IDs 0+10 on new recordings")
    dual.add_argument("--video-dir", type=Path, default=Path("video_data/double_marker"))
    dual.add_argument("--calibration", type=Path, default=Path("outputs/iphone_calibration.json"))
    dual.add_argument("--output-dir", type=Path,
                      help="separate output directory for non-default mounts")
    dual.add_argument("--marker10-mm", type=float, default=22.0,
                      help="measured outer black-square side of marker 10")
    dual.add_argument("--tip-x-mm", type=float,
                      help="marker 0 to tip X offset; inferred for known recording-batch directories")
    dual.add_argument("--reuse-tracking-from", type=Path,
                      help="reuse raw tracking from an earlier dual evaluation")
    evaluate_policy = commands.add_parser(
        "evaluate", help="run a trained policy closed loop from a set of start states")
    evaluate_policy.add_argument("checkpoint", help="pretrained_model folder")
    evaluate_policy.add_argument("--starts", choices=["excluded", "holdout", "dataset"],
                                 default="excluded",
                                 help="excluded: 55 starts from recordings never trained on "
                                      "(default); holdout: the 8 held-out clips; dataset: all 72")
    evaluate_policy.add_argument("--seeds", type=int, nargs="+", default=[0])
    evaluate_policy.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    evaluate_policy.add_argument("--video", action="store_true", help="save a video per rollout")
    download = commands.add_parser("download", help="fetch the published data into data/")
    download.add_argument("--raw", action="store_true", help="also the 134 iPhone recordings (6.8 GB)")
    download.add_argument("--policy", action="store_true", help="also the trained SmolVLA policy")
    sim_data = commands.add_parser("export-sim-dataset", help="capture two-camera scripted Panda episodes in LeRobot")
    sim_data.add_argument("--output-dir", type=Path, default=Path("outputs/datasets/usb_sim_two_cam"))
    sim_data.add_argument("--episodes", type=int, default=2)
    sim_data.add_argument("--seed", type=int, default=42)
    validate_data = commands.add_parser("validate-sim-dataset", help="read back the LeRobot dataset")
    validate_data.add_argument("--root", type=Path, default=Path("outputs/datasets/usb_sim_two_cam"))
    args = parser.parse_args()
    if args.command == "sim-demo":
        result = run_sim_demo(args.output)
        print(json.dumps(result, indent=2))
        if not result["alignment_reached"] or not result["insertion_reached"]:
            raise SystemExit("The predefined path did not reach the socket")
    elif args.command == "sim-view":
        run_viewer()
    elif args.command in ("calibrate", "calibrate-camera"):
        from .vision.calibrate import calibrate
        result = calibrate(args.input, args.output, square_m=args.square_mm / 1000,
                           marker_m=args.marker_mm / 1000, stride=args.stride)
        print(json.dumps({"output": str(args.output), "accepted_views": len(result["accepted_views"]),
                          "rms_reprojection_px": result["rms_reprojection_px"],
                          "board_center_span_fraction_xy": result["board_center_span_fraction_xy"],
                          "split_focal_difference_fraction": result["split_focal_difference_fraction"],
                          "warnings": result["warnings"]}, indent=2))
    elif args.command in ("track", "track-video"):
        from .vision.track import track_video
        output_dir = args.output_dir or Path("outputs/tracking") / Path(args.input).stem
        result = track_video(args.input, args.calibration, output_dir,
                             usb_marker_m=args.usb_marker_mm / 1000,
                             port_marker_m=args.port_marker_mm / 1000,
                             max_frames=args.max_frames)
        print(json.dumps(result, indent=2))
    elif args.command == "vision-smoke":
        from .vision.synthetic import run_smoke
        print(json.dumps(run_smoke(args.output_dir), indent=2))
    elif args.command == "inspect-marker-frames":
        from .vision.target_frames import inspect_frame
        print(json.dumps(inspect_frame(args.trajectory, args.calibration,
                                       args.frame_index, args.output,
                                       tip_x_mm=args.tip_x_mm), indent=2))
    elif args.command in ("replay", "replay-tracking"):
        from .retarget import replay_recording
        from .vision.target_frames import marker_tip_with_x_offset
        output_dir = args.output_dir or args.trajectory.parent / "replay"
        print(json.dumps(replay_recording(args.trajectory, output_dir,
                                          start_frame=args.start_frame,
                                          end_frame=args.end_frame,
                                          complete_segment=args.complete_segment,
                                          marker_tip=marker_tip_with_x_offset(args.tip_x_mm)), indent=2))
    elif args.command == "evaluate-recordings":
        from .dataset.tracking.recordings import evaluate_recordings
        print(json.dumps(evaluate_recordings(args.video_dir, args.calibration,
                                             args.output_dir,
                                             reuse_existing_replays=args.reuse_existing_replays,
                                             retrack=args.retrack), indent=2))
    elif args.command == "evaluate-dual-recordings":
        from .dataset.tracking.dual_recordings import evaluate_dual_recordings
        print(json.dumps(evaluate_dual_recordings(args.video_dir, args.calibration,
                                                  args.output_dir,
                                                  marker10_mm=args.marker10_mm,
                                                  tip_x_mm=args.tip_x_mm,
                                                  reuse_tracking_from=args.reuse_tracking_from), indent=2))
    elif args.command == "evaluate":
        import subprocess
        import sys
        downloaded = Path("data/mimicvla-panda-usb-15x7")
        dataset = downloaded if downloaded.exists() else Path("outputs/datasets/usb_15x7_no_funnel_full_v2")
        excluded = (downloaded / "eval_starts/excluded55" if downloaded.exists()
                    else Path("outputs/eval_sets/excluded_starts_v1"))
        start_dir, split = {"excluded": (excluded, "all"), "holdout": (dataset, "holdout"),
                            "dataset": (dataset, "all")}[args.starts]
        for seed in args.seeds:
            subprocess.run([sys.executable, "-m", "mimicvla.training.evaluate", args.checkpoint,
                            "--dataset-dir", str(start_dir), "--split", split, "--seed", str(seed),
                            "--output-dir", str(args.output_dir / f"{args.starts}_seed{seed}"),
                            *([] if args.video else ["--no-video"])], check=True)
    elif args.command == "download":
        from .dataset.publish.hub import download
        print(json.dumps(download(raw=args.raw, policy=args.policy), indent=2))
    elif args.command == "export-sim-dataset":
        from .dataset.lerobot import export_sim_dataset
        print(json.dumps(export_sim_dataset(args.output_dir, args.episodes, args.seed), indent=2))
    elif args.command == "validate-sim-dataset":
        from .dataset.lerobot import validate_sim_dataset
        print(json.dumps(validate_sim_dataset(args.root), indent=2))


if __name__ == "__main__":
    main()
