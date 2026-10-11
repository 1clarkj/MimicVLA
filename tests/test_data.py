"""Focused checks for LeRobot serialization and training image transforms."""

import json
import subprocess

import av
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import torch

from mimicvla.data.augmentation import appearance_augmentation_config
from mimicvla.data.raw_export import _policy_screen, export_raw_dataset
from mimicvla.data.sim_export import REPO_ID, TASK, validate_sim_dataset
from mimicvla.evaluation.recordings import fill_short_gaps


def test_appearance_augmentation_preserves_image_geometry():
    from lerobot.transforms import ImageTransforms

    config = appearance_augmentation_config()
    assert config.enable
    assert "affine" not in config.tfs
    rgb = torch.full((3, 256, 256), 0.5)
    augmented = ImageTransforms(config)(rgb)
    assert augmented.shape == rgb.shape
    assert float(augmented.min()) >= 0
    assert float(augmented.max()) <= 1


def test_policy_screen_explains_exclusions():
    row = {"eligible": True, "completed_segment": True, "joint_limits_ok": True,
           "bad_collision_steps": 0, "insertion_reached": True, "wall_contact_steps": 0}
    assert _policy_screen(row) == []
    row["wall_contact_steps"] = 4
    assert _policy_screen(row) == ["wall_contact"]
    row["eligible"] = False
    assert _policy_screen(row) == ["incomplete_tracked_insertion_span"]


def test_two_camera_lerobot_roundtrip(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "dataset"
    features = {
        "observation.state": {"dtype": "float32", "shape": (9,), "names": None},
        "action": {"dtype": "float32", "shape": (6,), "names": None},
        **{f"observation.images.camera{i}": {"dtype": "video", "shape": (256, 256, 3),
            "names": ["height", "width", "channel"]} for i in (1, 2)},
    }
    dataset = LeRobotDataset.create(REPO_ID, 20, features, root=root,
                                    robot_type="franka_panda_usb", video_backend="pyav")
    for frame_index in range(2):
        image = np.full((256, 256, 3), 50 + frame_index * 25, dtype=np.uint8)
        dataset.add_frame({"observation.state": np.zeros(9, dtype=np.float32),
                           "action": np.zeros(6, dtype=np.float32),
                           "observation.images.camera1": image,
                           "observation.images.camera2": image,
                           "task": TASK})
    dataset.save_episode()
    dataset.finalize()
    result = validate_sim_dataset(root)
    assert result["episodes"] == 1
    assert result["frames"] == 2


def test_raw_export_keeps_observed_and_interpolated_poses_separate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "video_data/20mm_marker/example.mov"
    source.parent.mkdir(parents=True)
    silent = tmp_path / "silent.mov"
    with imageio.get_writer(silent, fps=20, codec="libx264") as writer:
        for index in range(3):
            writer.append_data(np.full((64, 64, 3), 40 + 30 * index, dtype=np.uint8))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                    "-nostdin", "-y", "-i", str(silent), "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=0.15", "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", str(source)], check=True)
    evaluation = tmp_path / "evaluation"
    tracking = evaluation / "20mm_example/tracking"
    tracking.mkdir(parents=True)
    left = np.eye(4)
    right = np.eye(4)
    right[0, 3] = .01
    records = [
        {"quality": "ok", "frame_index": 0, "timestamp_s": 0.0,
         "T_port_usb_marker": left.tolist()},
        {"quality": "missing_usb", "frame_index": 1, "timestamp_s": .05,
         "T_port_usb_marker": None},
        {"quality": "ok", "frame_index": 2, "timestamp_s": .1,
         "T_port_usb_marker": right.tolist()},
    ]
    filled, count = fill_short_gaps(records)
    assert count == 1
    for name, rows in (("trajectory.jsonl", records), ("trajectory_filled.jsonl", filled)):
        (tracking / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (evaluation / "results.json").write_text(json.dumps([{
        "video": str(source), "usb_marker_mm": 20, "frames": 3,
        "valid_poses": 2, "valid_fraction": 2 / 3, "interpolated_frames": 1,
        "status": "insufficient_tracking_coverage", "eligible": False,
        "success": False, "contact_free_success": False,
        "reason": "too short for insertion",
    }]))
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    output = tmp_path / "raw"
    result = export_raw_dataset(evaluation, calibration, output)
    assert result["videos"] == 1
    assert result["new_videos_in_this_export"] == 1
    assert result["valid_relative_poses"] == 2
    assert result["processed_valid_relative_poses"] == 3
    assert result["videos_with_audio"] == 0
    with av.open(str(source)) as container:
        assert len(container.streams.audio) == 1  # Original is untouched.
    with av.open(str(output / "videos/20mm_example.mov")) as container:
        assert len(container.streams.audio) == 0
        assert len(container.streams.video) == 1
    raw = [json.loads(line) for line in (output / "poses/20mm_example.jsonl").read_text().splitlines()]
    processed = [json.loads(line) for line in
                 (output / "poses_processed/20mm_example.jsonl").read_text().splitlines()]
    assert raw[1]["T_socket_usb_tip"] is None
    assert processed[1]["pose_source"] == "interpolated"
    assert processed[1]["T_socket_usb_tip"] is not None
    metadata = json.loads((output / "videos/metadata.jsonl").read_text())
    assert metadata["policy_training_status"] == "excluded"
    assert metadata["policy_exclusion_reasons"] == ["incomplete_tracked_insertion_span"]
    assert export_raw_dataset(evaluation, calibration, output)["new_videos_in_this_export"] == 0
