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
    calibration = commands.add_parser("calibrate-camera", help="calibrate from ChArUco photos or video")
    calibration.add_argument("--input", type=Path, required=True, help="photo directory or video file")
    calibration.add_argument("--output", type=Path, default=Path("outputs/iphone_calibration.json"))
    calibration.add_argument("--stride", type=int, default=15, help="use every Nth photo/frame")
    calibration.add_argument("--square-mm", type=float, default=30.0)
    calibration.add_argument("--marker-mm", type=float, default=22.0)
    tracking = commands.add_parser("track-video", help="record relative USB/port marker poses")
    tracking.add_argument("--input", required=True, help="video file, or camera index such as 0")
    tracking.add_argument("--calibration", type=Path, required=True)
    tracking.add_argument("--output-dir", type=Path, default=Path("outputs/tracking"))
    tracking.add_argument("--usb-marker-mm", type=float, required=True,
                          help="measured outer black side: 12, 15, 18 or 20 mm on supplied sheet")
    tracking.add_argument("--port-marker-mm", type=float, default=50.0)
    tracking.add_argument("--max-frames", type=int, default=None)
    smoke = commands.add_parser("vision-smoke", help="generate and track synthetic marker video")
    smoke.add_argument("--output-dir", type=Path, default=Path("outputs/vision_smoke"))
    inspect = commands.add_parser("inspect-marker-frames", help="project provisional USB and socket target frames")
    inspect.add_argument("--trajectory", type=Path, required=True)
    inspect.add_argument("--calibration", type=Path, required=True)
    inspect.add_argument("--frame-index", type=int, default=90)
    inspect.add_argument("--output", type=Path, default=Path("outputs/target_frame_inspection.jpg"))
    replay = commands.add_parser("replay-tracking", help="retarget tracked USB motion and replay it on Panda")
    replay.add_argument("--trajectory", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, default=Path("outputs/replay_tracking"))
    replay.add_argument("--start-frame", type=int)
    replay.add_argument("--end-frame", type=int)
    replay.add_argument("--complete-segment", action="store_true",
                        help="continue through all recorded targets after first insertion success")
    evaluate = commands.add_parser("evaluate-recordings", help="track and replay all insertion videos")
    evaluate.add_argument("--video-dir", type=Path, default=Path("video_data"))
    evaluate.add_argument("--calibration", type=Path, default=Path("outputs/iphone_calibration.json"))
    evaluate.add_argument("--output-dir", type=Path, default=Path("outputs/video_evaluation"))
    evaluate.add_argument("--reuse-existing-replays", action="store_true",
                          help="reuse completed replay summaries when selected frame range matches")
    raw = commands.add_parser("export-raw-dataset", help="package videos and raw tracked poses for Hugging Face")
    raw.add_argument("--evaluation-dir", type=Path, default=Path("outputs/video_evaluation"))
    raw.add_argument("--calibration", type=Path, default=Path("outputs/iphone_calibration.json"))
    raw.add_argument("--output-dir", type=Path, default=Path("outputs/datasets/usb_marker_raw"))
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
    elif args.command == "calibrate-camera":
        from .vision.calibrate import calibrate
        result = calibrate(args.input, args.output, square_m=args.square_mm / 1000,
                           marker_m=args.marker_mm / 1000, stride=args.stride)
        print(json.dumps({"output": str(args.output), "accepted_views": len(result["accepted_views"]),
                          "rms_reprojection_px": result["rms_reprojection_px"],
                          "board_center_span_fraction_xy": result["board_center_span_fraction_xy"],
                          "split_focal_difference_fraction": result["split_focal_difference_fraction"],
                          "warnings": result["warnings"]}, indent=2))
    elif args.command == "track-video":
        from .vision.track import track_video
        result = track_video(args.input, args.calibration, args.output_dir,
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
                                       args.frame_index, args.output), indent=2))
    elif args.command == "replay-tracking":
        from .retarget import replay_recording
        print(json.dumps(replay_recording(args.trajectory, args.output_dir,
                                          start_frame=args.start_frame,
                                          end_frame=args.end_frame,
                                          complete_segment=args.complete_segment), indent=2))
    elif args.command == "evaluate-recordings":
        from .evaluation.recordings import evaluate_recordings
        print(json.dumps(evaluate_recordings(args.video_dir, args.calibration,
                                             args.output_dir,
                                             reuse_existing_replays=args.reuse_existing_replays), indent=2))
    elif args.command == "export-raw-dataset":
        from .data.raw_export import export_raw_dataset
        print(json.dumps(export_raw_dataset(args.evaluation_dir, args.calibration,
                                            args.output_dir), indent=2))
    elif args.command == "export-sim-dataset":
        from .data.sim_export import export_sim_dataset
        print(json.dumps(export_sim_dataset(args.output_dir, args.episodes, args.seed), indent=2))
    elif args.command == "validate-sim-dataset":
        from .data.sim_export import validate_sim_dataset
        print(json.dumps(validate_sim_dataset(args.root), indent=2))


if __name__ == "__main__":
    main()
