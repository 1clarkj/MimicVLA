"""Small Gym-like API for the Panda USB insertion task."""

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .controller import CartesianController, pose_from_site
from .scene import SOCKET_DEPTH_M, USB_TIP_Z_IN_HAND_M, SocketPose, build_model


@dataclass(frozen=True)
class SimConfig:
    socket: SocketPose = field(default_factory=SocketPose)
    control_hz: int = 20
    image_size: tuple[int, int] = (256, 256)
    max_steps: int = 200
    alignment_position_tolerance_m: float = 0.008
    alignment_orientation_tolerance_rad: float = np.deg2rad(10)
    insertion_depth_m: float = 0.006
    initial_ee_offset: tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)
    approach_clearance_m: float = 0.09


class PandaUSBEnv:
    """Action is a 6D world-frame Cartesian displacement per 20 Hz step."""

    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()
        self.model = build_model(self.config.socket)
        self.data = mujoco.MjData(self.model)
        self.control_dt = 1.0 / self.config.control_hz
        self.substeps = round(self.control_dt / self.model.opt.timestep)
        if not np.isclose(self.substeps * self.model.opt.timestep, self.control_dt):
            raise ValueError("control period must be a multiple of MuJoCo timestep")
        self.controller = CartesianController(self.model, self.data, self.control_dt)
        self._renderer: mujoco.Renderer | None = None
        self.steps = 0
        self.reset()

    def reset(self) -> dict:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[7] = 0
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset()
        socket_pos, socket_rot = pose_from_site(self.data, "socket_target")
        # Start with the connector horizontal and already facing the socket.
        # This small bias compensates position-actuator sag for the default pose.
        target_pos = (socket_pos - socket_rot[:, 2] *
                      (USB_TIP_Z_IN_HAND_M + self.config.approach_clearance_m) +
                      np.array([0.0, -0.0015, 0.0085]))
        offset = np.asarray(self.config.initial_ee_offset, dtype=float)
        target_pos += offset[:3]
        target_rot = Rotation.from_rotvec(offset[3:]).as_matrix() @ socket_rot
        self.controller.set_target(target_pos, target_rot)
        for _ in range(160):
            self.controller.update_actuators()
            mujoco.mj_step(self.model, self.data, self.substeps)
        self.data.time = 0.0
        self.controller.reset()
        self.steps = 0
        return self.observation(render=False)

    def step(self, action: np.ndarray) -> tuple[dict, bool, dict]:
        self.controller.apply_displacement(action)
        self.controller.update_actuators()
        mujoco.mj_step(self.model, self.data, self.substeps)
        self.steps += 1
        info = self.metrics()
        done = info["insertion_success"] or self.steps >= self.config.max_steps
        return self.observation(render=False), done, info

    def render(self, camera: str = "front_rgb") -> np.ndarray:
        if self._renderer is None:
            height, width = self.config.image_size
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def observation(self, render: bool = True) -> dict:
        state = np.array([self.data.qpos[self.model.joint(name).qposadr[0]] for name in
                          [*(f"joint{i}" for i in range(1, 8)), "finger_joint1", "finger_joint2"]], dtype=np.float32)
        return {
            "timestamp": float(self.data.time),
            "observation.state": state,
            "observation.images.camera1": self.render("front_rgb") if render else None,
            "observation.images.camera2": self.render("wrist_rgb") if render else None,
        }

    def metrics(self) -> dict:
        tip_pos, tip_rot = pose_from_site(self.data, "usb_tip")
        socket_pos, socket_rot = pose_from_site(self.data, "socket_target")
        delta_local = socket_rot.T @ (tip_pos - socket_pos)
        orientation_error = Rotation.from_matrix(socket_rot.T @ tip_rot).magnitude()
        radial_error = float(np.linalg.norm(delta_local[:2]))
        depth = float(delta_local[2])
        aligned = (radial_error < self.config.alignment_position_tolerance_m
                   and orientation_error < self.config.alignment_orientation_tolerance_rad
                   and abs(depth) < 0.012)
        inserted = (radial_error < 0.0015
                    and orientation_error < np.deg2rad(5)
                    and self.config.insertion_depth_m <= depth <= SOCKET_DEPTH_M)
        socket_id = self.model.body("socket").id
        bad_collision = False
        wall_contacts = 0
        socket_contacts = 0
        max_wall_penetration = 0.0
        max_wall_force = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body_ids = (
                self.model.geom(contact.geom1).bodyid[0],
                self.model.geom(contact.geom2).bodyid[0],
            )
            if socket_id in body_ids and body_ids[0] != body_ids[1]:
                # MuJoCo contacts are soft; count penetration beyond 1 mm as a strike.
                bad_collision |= contact.dist < -0.001
                if contact.dist <= 0:
                    socket_contacts += 1
                    wall_geom = any(self.model.geom(geom_id).name.startswith("wall_")
                                    for geom_id in (contact.geom1, contact.geom2))
                    if wall_geom:
                        wall_contacts += 1
                        max_wall_penetration = max(max_wall_penetration, -float(contact.dist))
                        force = np.zeros(6)
                        mujoco.mj_contactForce(self.model, self.data, i, force)
                        max_wall_force = max(max_wall_force, float(force[0]))
        return {
            "radial_error_m": radial_error,
            "axial_error_m": depth,
            "orientation_error_rad": float(orientation_error),
            "alignment_success": bool(aligned),
            "insertion_success": bool(inserted and not bad_collision),
            "bad_collision": bool(bad_collision),
            "socket_contact_count": socket_contacts,
            "wall_contact_count": wall_contacts,
            "wall_contact": bool(wall_contacts),
            "max_wall_penetration_m": max_wall_penetration,
            "max_wall_normal_force_n": max_wall_force,
        }

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
