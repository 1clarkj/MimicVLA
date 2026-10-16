"""Build the USB task on the official MuJoCo Menagerie Panda scene."""

from dataclasses import dataclass

import mujoco
import mujoco_menagerie as menagerie
import numpy as np
from scipy.spatial.transform import Rotation


USB_TIP_Z_IN_HAND_M = 0.16
USB_HALF_WIDTH_M = 0.006
USB_HALF_HEIGHT_M = 0.00225
SOCKET_DEPTH_M = 0.018
SOCKET_INNER_HALF_X_M = USB_HALF_WIDTH_M + 0.0007
SOCKET_INNER_HALF_Y_M = USB_HALF_HEIGHT_M + 0.0007


@dataclass(frozen=True)
class SocketPose:
    """World pose of a vertical socket entrance; local +Z points inward."""

    position_m: tuple[float, float, float] = (0.65, 0.0, 0.60)
    # Local +X is world -Y (long side), +Y is world -Z (short side),
    # and +Z is world +X (the horizontal insertion direction).
    quaternion_wxyz: tuple[float, float, float, float] = (-0.5, 0.5, -0.5, 0.5)


@dataclass(frozen=True)
class SocketGeometry:
    """Physical cavity and optional entry guide; dimensions are half widths."""

    inner_half_x_m: float = SOCKET_INNER_HALF_X_M
    inner_half_y_m: float = SOCKET_INNER_HALF_Y_M
    mouth_half_x_m: float = SOCKET_INNER_HALF_X_M
    mouth_half_y_m: float = SOCKET_INNER_HALF_Y_M
    funnel_depth_m: float = 0.0
    funnel_solref_timeconst_s: float = 0.02


STRICT_SOCKET = SocketGeometry()
TRAINING_SOCKET = SocketGeometry(inner_half_x_m=.009,
                                 inner_half_y_m=.005,
                                 mouth_half_x_m=.015,
                                 mouth_half_y_m=.010,
                                 funnel_depth_m=.008)


def curriculum_socket(tightness: float) -> SocketGeometry:
    """Interpolate from the guided training port (0) to the strict port (1)."""
    if not 0 <= tightness <= 1:
        raise ValueError("tightness must be between 0 and 1")
    return SocketGeometry(**{
        field: (1 - tightness) * getattr(TRAINING_SOCKET, field) +
               tightness * getattr(STRICT_SOCKET, field)
        for field in SocketGeometry.__dataclass_fields__
    })


def _camera_quat_wxyz(position: np.ndarray, target: np.ndarray,
                      up_hint: np.ndarray | None = None) -> np.ndarray:
    """MuJoCo cameras look along local -Z, with local +Y as up."""
    back = position - target
    back /= np.linalg.norm(back)
    right = np.cross(np.array([0.0, 0.0, 1.0]) if up_hint is None else up_hint, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    xyzw = Rotation.from_matrix(np.column_stack((right, up, back))).as_quat()
    return xyzw[[3, 0, 1, 2]]


def build_model(socket: SocketPose, geometry: SocketGeometry = STRICT_SOCKET) -> mujoco.MjModel:
    """Return an editable official Panda scene with a simple USB-A fixture."""
    if (geometry.inner_half_x_m <= USB_HALF_WIDTH_M or
            geometry.inner_half_y_m <= USB_HALF_HEIGHT_M or
            geometry.mouth_half_x_m < geometry.inner_half_x_m or
            geometry.mouth_half_y_m < geometry.inner_half_y_m or
            geometry.funnel_depth_m < 0 or
            geometry.funnel_solref_timeconst_s <= 0):
        raise ValueError("Socket geometry must clear the plug and widen toward its mouth")
    spec = menagerie.get("franka_emika_panda").spec("scene")
    hand = spec.body("hand")
    # Sites are geometric references for control and scoring, not policy cues.
    hand.add_site(name="ee_site", pos=[0, 0, 0], size=[0.004], rgba=[0, 0.8, 1, 0])
    hand.add_geom(
        name="usb_handle", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0.105], size=[0.004, 0.002, 0.04],
        rgba=[0.14, 0.15, 0.18, 1], contype=0, conaffinity=0,
    )
    hand.add_geom(
        name="usb_plug",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0.145],
        size=[USB_HALF_WIDTH_M, USB_HALF_HEIGHT_M, 0.015],
        rgba=[0.75, 0.76, 0.8, 1],
        density=7800,
    )
    hand.add_site(
        name="usb_tip",
        pos=[0, 0, USB_TIP_Z_IN_HAND_M],
        size=[0.002],
        rgba=[1, 0.2, 0.1, 0],
    )
    wrist_pos = np.array([0.0, -0.045, 0.070])
    wrist_target = np.array([0.0, 0.0, 0.160])
    hand.add_camera(
        name="wrist_rgb", pos=wrist_pos,
        quat=_camera_quat_wxyz(wrist_pos, wrist_target, np.array([0.0, -1.0, 0.0])),
        fovy=70,
    )

    body = spec.worldbody.add_body(
        name="socket", pos=list(socket.position_m), quat=list(socket.quaternion_wxyz)
    )
    body.add_site(name="socket_target", pos=[0, 0, 0], size=[0.003], rgba=[0.1, 1, 0.1, 0])
    # Four walls leave a slightly oversized cavity open at local Z = 0.
    inner_x = geometry.inner_half_x_m
    inner_y = geometry.inner_half_y_m
    wall = 0.003
    for sign in (-1, 1):
        body.add_geom(
            name=f"socket_x_{sign}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[sign * (inner_x + wall / 2), 0, SOCKET_DEPTH_M / 2],
            size=[wall / 2, inner_y + wall, SOCKET_DEPTH_M / 2],
            rgba=[0.12, 0.16, 0.2, 1],
        )
        body.add_geom(
            name=f"socket_y_{sign}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0, sign * (inner_y + wall / 2), SOCKET_DEPTH_M / 2],
            size=[inner_x, wall / 2, SOCKET_DEPTH_M / 2],
            rgba=[0.12, 0.16, 0.2, 1],
        )
    body.add_geom(
        name="socket_back", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, SOCKET_DEPTH_M + wall / 2],
        size=[inner_x + wall, inner_y + wall, wall / 2],
        rgba=[0.12, 0.16, 0.2, 1],
    )
    # Laptop-like front face. Four boxes form a real opening; their front
    # surfaces and the socket mouth are both at local Z = 0. Local +Z goes
    # inside the laptop. Unlike the previous block, the socket does not stick
    # out in front of the wall.
    panel_half_x = 0.12
    panel_half_y = 0.08
    panel_depth = 0.008
    side_width = (panel_half_x - inner_x) / 2
    top_height = (panel_half_y - inner_y) / 2
    for sign in (-1, 1):
        body.add_geom(
            name=f"wall_side_{sign}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[sign * (inner_x + side_width), 0, panel_depth / 2],
            size=[side_width, inner_y, panel_depth / 2],
            rgba=[0.10, 0.11, 0.13, 1],
        )
        body.add_geom(
            name=f"wall_top_{sign}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0, sign * (inner_y + top_height), panel_depth / 2],
            size=[panel_half_x, top_height, panel_depth / 2],
            rgba=[0.10, 0.11, 0.13, 1],
        )
    if geometry.funnel_depth_m:
        depth = geometry.funnel_depth_m
        mouth_x, mouth_y = geometry.mouth_half_x_m, geometry.mouth_half_y_m
        for sign in (-1, 1):
            x_delta = inner_x - mouth_x
            x_angle = np.arctan2(sign * x_delta, depth)
            x_quat = Rotation.from_euler("y", x_angle).as_quat()[[3, 0, 1, 2]]
            body.add_geom(
                name=f"funnel_x_{sign}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[sign * (inner_x + mouth_x) / 2, 0, -depth / 2],
                quat=x_quat.tolist(),
                size=[wall / 2, mouth_y + wall, np.hypot(depth, x_delta) / 2],
                solref=[geometry.funnel_solref_timeconst_s, 1.0],
                rgba=[0.25, 0.27, 0.30, 1],
            )
            y_delta = inner_y - mouth_y
            y_angle = np.arctan2(-sign * y_delta, depth)
            y_quat = Rotation.from_euler("x", y_angle).as_quat()[[3, 0, 1, 2]]
            body.add_geom(
                name=f"funnel_y_{sign}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0, sign * (inner_y + mouth_y) / 2, -depth / 2],
                quat=y_quat.tolist(),
                size=[mouth_x + wall, wall / 2, np.hypot(depth, y_delta) / 2],
                solref=[geometry.funnel_solref_timeconst_s, 1.0],
                rgba=[0.25, 0.27, 0.30, 1],
            )
    # A non-colliding metallic rim makes the flush slot visible to RGB cameras.
    # Its sub-millimetre thickness is visual only; the panel boxes own contact.
    rim = 0.0012
    for sign in (-1, 1):
        body.add_geom(
            name=f"rim_side_{sign}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[sign * (inner_x + rim / 2), 0, -0.0001],
            size=[rim / 2, inner_y + rim, 0.0001],
            rgba=[0.42, 0.44, 0.46, 1], contype=0, conaffinity=0,
        )
        body.add_geom(
            name=f"rim_top_{sign}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0, sign * (inner_y + rim / 2), -0.0001],
            size=[inner_x + rim, rim / 2, 0.0001],
            rgba=[0.42, 0.44, 0.46, 1], contype=0, conaffinity=0,
        )

    # View the opening from the approach side; this is a provisional policy camera.
    cam_pos = np.array([0.52, -0.28, 0.72])
    cam_target = np.array(socket.position_m)
    spec.worldbody.add_camera(
        name="front_rgb", pos=cam_pos, quat=_camera_quat_wxyz(cam_pos, cam_target), fovy=30
    )
    return spec.compile()
