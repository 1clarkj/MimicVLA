"""Cartesian target tracking with Mink differential IK."""

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def pose_from_site(data: mujoco.MjData, site_name: str) -> tuple[np.ndarray, np.ndarray]:
    site = data.site(site_name)
    return site.xpos.copy(), site.xmat.reshape(3, 3).copy()


class CartesianController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, control_dt: float):
        self.model = model
        self.data = data
        self.control_dt = control_dt
        self.configuration = mink.Configuration(model)
        self.task = mink.FrameTask("ee_site", "site", position_cost=1.0, orientation_cost=0.3)
        self.target_position = np.zeros(3)
        self.target_rotation = np.eye(3)

    def reset(self) -> None:
        self.configuration.update(self.data.qpos)
        self.target_position, self.target_rotation = pose_from_site(self.data, "ee_site")

    def set_target(self, position_m: np.ndarray, rotation: np.ndarray) -> None:
        self.target_position = np.asarray(position_m, dtype=float).copy()
        self.target_rotation = np.asarray(rotation, dtype=float).copy()

    def apply_displacement(self, action: np.ndarray) -> None:
        """Six values: world XYZ metres, then world-axis rotation vector radians."""
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("action must contain six finite values")
        self.target_position += action[:3]
        self.target_rotation = Rotation.from_rotvec(action[3:]).as_matrix() @ self.target_rotation

    def update_actuators(self) -> None:
        self.configuration.update(self.data.qpos)
        self.task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(self.target_rotation), self.target_position
            )
        )
        velocity = mink.solve_ik(self.configuration, [self.task], self.control_dt, "daqp")
        self.configuration.integrate_inplace(velocity, self.control_dt)
        self.data.ctrl[:7] = self.configuration.q[:7]
        self.data.ctrl[7] = 0.0  # Menagerie Panda: 0 closed, 255 open.
