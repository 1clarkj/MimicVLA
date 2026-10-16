"""Historical replay metrics can be rescored without changing the source run."""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from mimicvla.evaluation.rescore import rescore_replays
from mimicvla.sim.scene import SocketPose, USB_TIP_Z_IN_HAND_M


def test_rescore_requires_measured_twelve_millimetres(tmp_path):
    batch = tmp_path / "old_batch"
    replay = batch / "20mm_demo" / "replay"
    replay.mkdir(parents=True)
    (batch / "results.json").write_text(json.dumps([
        {"video": "video_data/20mm_marker/demo.mov", "usb_marker_mm": 20,
         "eligible": True}]))
    def metric(depth):
        return {"radial_error_m": .001, "orientation_error_rad": .15,
                "socket_lateral_x_m": .001, "socket_lateral_y_m": 0,
                "axial_error_m": depth, "bad_collision": False,
                "wall_contact": False}
    (replay / "retargeted_replay.json").write_text(json.dumps([
        {"metrics": metric(.011)}, {"metrics": metric(.012)}]))
    (replay / "summary.json").write_text(json.dumps({
        "bad_collision_steps": 0, "wall_contact_steps": 0,
        "completed_segment": True, "all_joint_limits_ok": True}))
    output = tmp_path / "rescore"
    summary = rescore_replays(output, {"test": batch})
    assert summary["required_tip_depth_mm"] == 12
    assert summary["by_batch"]["test/single_marker"]["episode_successes"] == 1
    result = json.loads((output / "results.json").read_text())[0]
    assert result["contact_free_success"]
    (replay / "retargeted_replay.json").write_text(json.dumps([
        {"metrics": metric(.011)}, {"metrics": metric(.0119)}]))
    assert rescore_replays(output, {"test": batch})["by_batch"]["test/single_marker"]["episode_successes"] == 0


def test_rescore_old_replay_uses_tip_entry_despite_angle_and_contact(tmp_path):
    batch = tmp_path / "batch"
    replay = batch / "20mm_demo" / "replay"
    replay.mkdir(parents=True)
    (batch / "results.json").write_text(json.dumps([{
        "video": "video_data/20mm_marker/demo.mov", "usb_marker_mm": 20,
        "eligible": True}]))
    socket = SocketPose()
    quaternion = np.asarray(socket.quaternion_wxyz)
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    tip = np.asarray(socket.position_m) + rotation @ np.array([.002, .001, .014])
    ee = np.eye(4)
    ee[:3, :3] = rotation
    ee[:3, 3] = tip - rotation @ np.array([0, 0, USB_TIP_Z_IN_HAND_M])
    (replay / "retargeted_replay.json").write_text(json.dumps([{
        "T_world_ee_actual": ee.tolist(),
        "metrics": {"radial_error_m": float(np.hypot(.002, .001)),
                    "orientation_error_rad": np.deg2rad(9),
                    "axial_error_m": .014, "bad_collision": True},
    }]))
    (replay / "summary.json").write_text(json.dumps({
        "bad_collision_steps": 1, "wall_contact_steps": 1,
        "completed_segment": True, "all_joint_limits_ok": True}))
    rescore_replays(tmp_path / "scored", {"test": batch})
    row = json.loads((tmp_path / "scored" / "results.json").read_text())[0]
    assert row["episode_success"]
    assert not row["replay_quality_ok"]
