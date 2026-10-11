"""Integrated checks for the first Panda simulation milestone."""

import numpy as np
from scipy.spatial.transform import Rotation

from mimicvla.sim import PandaUSBEnv, SimConfig
from mimicvla.sim.controller import pose_from_site
from mimicvla.sim.demo import scripted_actions


def test_model_control_and_rgb_render():
    env = PandaUSBEnv()
    try:
        assert env.model.njnt == 9
        assert env.model.nu == 8
        image = env.render()
        assert image.shape == (256, 256, 3)
        assert image.dtype == np.uint8
        assert image.std() > 5
        wrist = env.render("wrist_rgb")
        assert wrist.shape == image.shape
        assert wrist.std() > 5
        assert not np.array_equal(wrist, image)
        start, _ = pose_from_site(env.data, "ee_site")
        for _ in range(30):
            env.step(np.array([0.0005, 0, 0, 0, 0, 0]))
        end, _ = pose_from_site(env.data, "ee_site")
        assert end[0] - start[0] > 0.008
        _, rotation_before = pose_from_site(env.data, "ee_site")
        for _ in range(30):
            env.step(np.array([0, 0, 0, 0, 0, 0.001]))
        _, rotation_after = pose_from_site(env.data, "ee_site")
        angular_delta = Rotation.from_matrix(rotation_after @ rotation_before.T).as_rotvec()
        assert angular_delta[2] > 0.02
        assert np.isfinite(env.observation(render=False)["observation.state"]).all()
    finally:
        env.close()


def test_predefined_path_reaches_alignment_and_insertion():
    env = PandaUSBEnv()
    try:
        tip, _ = pose_from_site(env.data, "usb_tip")
        socket, socket_rotation = pose_from_site(env.data, "socket_target")
        np.testing.assert_allclose(socket_rotation[:, 2], [1, 0, 0], atol=1e-6)
        assert abs(socket_rotation[1, 0]) > 0.99  # Long side is world Y.
        assert abs(socket_rotation[2, 1]) > 0.99  # Short side is world Z.
        assert tip[0] < socket[0]
        saw_alignment = False
        saw_insertion = False
        for action in scripted_actions(env):
            _, _, info = env.step(action)
            saw_alignment |= info["alignment_success"]
            saw_insertion |= info["insertion_success"]
        assert saw_alignment
        assert saw_insertion
        assert not info["bad_collision"]
    finally:
        env.close()


def test_reset_supports_initial_pose_offset():
    reference = PandaUSBEnv()
    shifted = PandaUSBEnv(SimConfig(initial_ee_offset=(0, 0.01, 0, 0, 0, 0)))
    try:
        p0, _ = pose_from_site(reference.data, "ee_site")
        p1, _ = pose_from_site(shifted.data, "ee_site")
        assert p1[1] - p0[1] > 0.005
    finally:
        reference.close()
        shifted.close()


def test_flush_wall_and_contact_reporting():
    env = PandaUSBEnv(SimConfig(initial_ee_offset=(0, .03, 0, 0, 0, 0), max_steps=100))
    try:
        for name in ("wall_side_-1", "wall_side_1", "wall_top_-1", "wall_top_1"):
            geom = env.model.geom(name)
            assert np.isclose(geom.pos[2] - geom.size[2], 0, atol=1e-9)
        infos = [env.step(np.array([.003, 0, 0, 0, 0, 0]))[2] for _ in range(60)]
        assert any(info["wall_contact"] for info in infos)
        assert max(info["max_wall_normal_force_n"] for info in infos) > 0
        assert not any(info["insertion_success"] for info in infos)
    finally:
        env.close()
