"""A simple, repeatable approach path for testing the environment."""

from collections.abc import Iterator

import numpy as np

from .controller import pose_from_site
from .env import PandaUSBEnv


def scripted_actions(env: PandaUSBEnv) -> Iterator[np.ndarray]:
    """Yield 20 Hz actions; the caller must step the environment between yields."""
    tip, _ = pose_from_site(env.data, "usb_tip")
    socket, socket_rotation = pose_from_site(env.data, "socket_target")
    insertion_axis = socket_rotation[:, 2]
    preinsert = socket - insertion_axis * 0.012
    approach_distance = float(np.dot(preinsert - tip, insertion_axis))
    for _ in range(60):
        yield np.r_[insertion_axis * (approach_distance / 60), np.zeros(3)]
    for _ in range(30):
        yield np.zeros(6)

    for _ in range(70):
        yield np.r_[insertion_axis * (0.020 / 70), np.zeros(3)]
    for _ in range(40):
        yield np.zeros(6)
