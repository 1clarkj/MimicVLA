"""Capture matched Panda state, Cartesian action, and two RGB policy views."""

import json
from pathlib import Path

import numpy as np

from ..sim.demo import scripted_actions
from ..sim.env import PandaUSBEnv, SimConfig
from ..sim.scene import SocketPose


TASK = "Insert the USB stick into the USB port."
REPO_ID = "mimicvla/usb_sim_two_cam"


def export_sim_dataset(output_dir: Path, episodes: int = 2, seed: int = 42) -> dict:
    """Write a small scripted baseline using the installed LeRobot dataset API.

    Actions and observations are paired before each control step. This function
    deliberately excludes contact-prone human replays until they are curated.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty dataset directory: {output_dir}")
    rng = np.random.default_rng(seed)
    features = {
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
    dataset = LeRobotDataset.create(repo_id=REPO_ID, fps=20, features=features,
                                    root=output_dir, robot_type="franka_panda_usb",
                                    use_videos=True, video_backend="pyav")
    manifest = []
    try:
        for index in range(episodes):
            # Modest socket shifts give a reproducible baseline without putting
            # the scripted start outside the narrow cavity.
            shift = rng.uniform(-0.002, 0.002, size=3)
            default = SocketPose()
            socket = SocketPose(tuple(np.asarray(default.position_m) + shift),
                                default.quaternion_wxyz)
            env = PandaUSBEnv(SimConfig(socket=socket))
            frames = 0
            wall_steps = 0
            success = False
            try:
                for action in scripted_actions(env):
                    obs = env.observation(render=True)
                    dataset.add_frame({
                        "observation.state": obs["observation.state"],
                        "observation.images.camera1": obs["observation.images.camera1"],
                        "observation.images.camera2": obs["observation.images.camera2"],
                        "action": np.asarray(action, dtype=np.float32),
                        "task": TASK,
                    })
                    _, _, info = env.step(action)
                    frames += 1
                    wall_steps += int(info["wall_contact"])
                    success |= info["insertion_success"]
                if not success:
                    raise ValueError(f"Episode {index}: insertion={success}, wall-contact steps={wall_steps}")
                dataset.save_episode()
                manifest.append({"episode_index": index, "frames": frames,
                                 "socket_position_m": list(socket.position_m),
                                 "insertion_success": success, "wall_contact_steps": wall_steps})
            finally:
                env.close()
    finally:
        dataset.finalize()
    summary = {"repo_id": REPO_ID, "root": str(output_dir), "episodes": len(manifest),
               "frames": sum(item["frames"] for item in manifest), "seed": seed,
               "required_tip_depth_m": SimConfig().insertion_depth_m,
               "source": "scripted MuJoCo baseline; no human replay episodes",
               "camera1": "fixed front_rgb", "camera2": "hand-mounted wrist_rgb",
               "state": "7 Panda arm joint radians + 2 finger metres, pre-action",
               "action": "world XYZ displacement metres + world rotation vector radians per 0.05 s",
               "episode_manifest": manifest}
    (output_dir / "capture_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "README.md").write_text("""---
tags:
- robotics
- lerobot
- simulation
---

# Panda USB insertion: two-camera simulation baseline

Local LeRobot dataset for the instruction **Insert the USB stick into the USB port.**
The two RGB streams are a fixed overview (`camera1`) and a hand-mounted wrist
view (`camera2`), each 256×256 at 20 Hz. Each row pairs a pre-action measured
Panda state (seven joint angles in radians, two finger positions in metres)
with the Cartesian displacement applied at that control step (world XYZ metres
and world-axis rotation vector radians). Gripper closure is fixed. Timestamps
and episode boundaries are generated by LeRobot. See `capture_manifest.json`
for seeds, socket poses, and contact checks.

These episodes come from a scripted MuJoCo controller, not the iPhone videos or
a learned policy. Socket positions vary by at most 2 mm. They test the dataset
pipeline only; they are too few and too similar to establish policy performance.
The physical task criterion requires at least 12 mm tip depth. The simulated
socket cavity is 18 mm deep, allowing a little commanded-motion margin.
No image augmentation is baked into the videos. Apply modest appearance changes
during training, and keep evaluation unaugmented. Review the resulting cameras
and choose a license before publishing this local dataset.
""")
    return summary


def validate_sim_dataset(root: Path) -> dict:
    """Read the finalized dataset and verify the policy inputs and boundaries."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=REPO_ID, root=root, video_backend="pyav")
    assert dataset.num_episodes >= 1
    assert len(dataset) >= dataset.num_episodes
    sample = dataset[0]
    assert tuple(sample["observation.state"].shape) == (9,)
    assert tuple(sample["action"].shape) == (6,)
    for key in ("observation.images.camera1", "observation.images.camera2"):
        assert tuple(sample[key].shape) == (3, 256, 256), (key, sample[key].shape)
    assert np.isfinite(sample["observation.state"].numpy()).all()
    assert np.isfinite(sample["action"].numpy()).all()
    assert sample["task"] == TASK
    assert float(sample["timestamp"]) == 0.0
    assert abs(float(dataset[1]["timestamp"]) - 0.05) < 1e-5
    checked_indices = sorted({*range(0, len(dataset), max(1, len(dataset) // 20)),
                              len(dataset) - 1})
    episode_ids = np.asarray([int(dataset[i]["episode_index"]) for i in checked_indices])
    assert episode_ids.min() == 0 and episode_ids.max() == dataset.num_episodes - 1
    return {"episodes": dataset.num_episodes, "frames": len(dataset),
            "state_shape": list(sample["observation.state"].shape),
            "action_shape": list(sample["action"].shape),
            "camera_shape_chw": list(sample["observation.images.camera1"].shape)}
