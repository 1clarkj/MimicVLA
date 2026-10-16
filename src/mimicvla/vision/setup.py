"""A recording setup: camera, ArUco markers and the two task transforms.

One JSON file describes everything that is specific to a physical rig:

    {
      "camera": "iphone_calibration.json",
      "port_marker": {"id": 1, "size_mm": 50},
      "usb_marker": {"id": 0, "size_mm": 20},
      "usb_marker_2": {"id": 10, "size_mm": 22, "T_usb_marker_usb_marker_2": [[...]]},
      "T_socket_port_marker": [[...]],
      "T_usb_marker_usb_tip": [[...]]
    }

T_A_B maps coordinates in frame B into frame A; lengths are metres. Marker
frames are OpenCV ArUco frames: origin at the marker centre, x right and y up
as printed, z out of the printed face. The socket frame has its origin at the
centre of the port opening, z pointing into the port and x along the long
side of the opening. The USB tip frame has its origin at the centre of the
plug's leading face, z along the insertion direction and x along the long side.
`camera` is relative to the setup file. `usb_marker_2` is optional: a second
marker on the plug that keeps the pose when the first one is hidden; its
transform can be fitted with `mimicvla fit-marker-mount`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Marker:
    id: int
    size_m: float


@dataclass
class Setup:
    camera: Path
    port_marker: Marker
    usb_marker: Marker
    T_socket_port_marker: np.ndarray
    T_usb_marker_usb_tip: np.ndarray
    usb_marker_2: Marker | None = None
    T_usb_marker_usb_marker_2: np.ndarray | None = None
    path: Path | None = None

    def socket_from_usb_marker(self, T_port_usb_marker: np.ndarray) -> np.ndarray:
        """T_socket_usb_tip from a tracked T_port_marker_usb_marker."""
        return self.T_socket_port_marker @ T_port_usb_marker @ self.T_usb_marker_usb_tip


def _marker(entry: dict) -> Marker:
    return Marker(int(entry["id"]), float(entry["size_mm"]) / 1000)


def _transform(value, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError(f"{name} must be a 4x4 homogeneous transform")
    if not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} must have an orthonormal rotation")
    return matrix


def load_setup(path: Path) -> Setup:
    path = Path(path)
    data = json.loads(path.read_text())
    second = data.get("usb_marker_2")
    setup = Setup(
        camera=path.parent / data["camera"],
        port_marker=_marker(data["port_marker"]),
        usb_marker=_marker(data["usb_marker"]),
        T_socket_port_marker=_transform(data["T_socket_port_marker"], "T_socket_port_marker"),
        T_usb_marker_usb_tip=_transform(data["T_usb_marker_usb_tip"], "T_usb_marker_usb_tip"),
        usb_marker_2=_marker(second) if second else None,
        T_usb_marker_usb_marker_2=(
            _transform(second["T_usb_marker_usb_marker_2"], "T_usb_marker_usb_marker_2")
            if second and second.get("T_usb_marker_usb_marker_2") is not None else None),
        path=path)
    ids = [setup.port_marker.id, setup.usb_marker.id] + ([second["id"]] if second else [])
    if len(set(ids)) != len(ids):
        raise ValueError("Every marker needs its own ArUco id")
    return setup
