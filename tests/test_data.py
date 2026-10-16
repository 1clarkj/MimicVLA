"""LeRobot serialization round trip for the two-camera Panda dataset."""

import numpy as np

from mimicvla.dataset.lerobot import REPO_ID, TASK, validate_sim_dataset


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
