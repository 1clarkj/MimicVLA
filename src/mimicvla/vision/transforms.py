"""Homogeneous transforms: T_A_B maps coordinates in B into A."""

import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_rotvec(np.asarray(rvec).reshape(3)).as_matrix()
    transform[:3, 3] = np.asarray(tvec).reshape(3)
    return transform


def inverse(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def translation_rotvec(transform: np.ndarray) -> tuple[list[float], list[float]]:
    return (
        np.asarray(transform[:3, 3]).tolist(),
        Rotation.from_matrix(transform[:3, :3]).as_rotvec().tolist(),
    )
