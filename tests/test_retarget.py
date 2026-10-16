"""Checks that marker-frame measurements become feasible controller targets."""

import numpy as np
import pytest

from scipy.spatial.transform import Rotation

from mimicvla.retarget import _densify, _matrix, lowpass_poses, resample_segment, valid_segment
from mimicvla.vision.target_frames import provisional_mounts
from mimicvla.vision.transforms import inverse


def test_retarget_resamples_and_preserves_final_insertion():
    port_mount, marker_tip = provisional_mounts()
    records = []
    for index, depth in enumerate([-.04, -.025, -.010, .008]):
        socket_tip = _matrix(np.array([0., 0., depth]), np.eye(3))
        port_usb = inverse(port_mount) @ socket_tip @ inverse(marker_tip)
        records.append({"frame_index": index, "timestamp_s": index * .1,
                        "quality": "ok", "T_port_usb_marker": port_usb.tolist()})
    times, targets, raw = resample_segment(valid_segment(records))
    assert len(times) == 7
    assert len(raw) == 4
    np.testing.assert_allclose(targets[-1][:3, 3], [0, 0, .008], atol=1e-12)
    start = _matrix(np.array([0., 0., -.08]), np.eye(3))
    commands = _densify(targets, start)
    assert len(commands) > len(targets)
    np.testing.assert_allclose(commands[-1], targets[-1], atol=1e-12)
    assert max(np.linalg.norm(b[:3, 3] - a[:3, 3]) for a, b in
               zip([start, *commands[:-1]], commands)) <= .003001


def test_retarget_rejects_marker_gap():
    records = [{"quality": "ok"}, {"quality": "missing_usb"}, {"quality": "ok"}]
    try:
        valid_segment(records, 0, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("A selected segment crossed missing marker data")


def test_resampling_does_not_exceed_final_timestamp():
    port_mount, marker_tip = provisional_mounts()
    start = 1.0833333333333335
    end = 3.8333333333333335
    records = []
    for index, timestamp in enumerate(np.linspace(start, end, 5)):
        socket_tip = _matrix(np.array([0., 0., -.02 + index * .005]), np.eye(3))
        port_usb = inverse(port_mount) @ socket_tip @ inverse(marker_tip)
        records.append({"frame_index": index, "timestamp_s": float(timestamp),
                        "quality": "ok", "T_port_usb_marker": port_usb.tolist()})
    times, _, _ = resample_segment(records)
    assert times[0] == start
    assert times[-1] == end
    assert np.all(times <= end)


def test_new_mount_x_offset_changes_tip_targets_without_changing_marker_track():
    port_mount, marker_tip = provisional_mounts()
    records = []
    for index, depth in enumerate([-.04, -.025, -.010, .008]):
        socket_tip = _matrix(np.array([0., 0., depth]), np.eye(3))
        port_usb = inverse(port_mount) @ socket_tip @ inverse(marker_tip)
        records.append({"frame_index": index, "timestamp_s": index * .1,
                        "quality": "ok", "T_port_usb_marker": port_usb.tolist()})
    _, baseline, _ = resample_segment(records)
    adjusted_mount = marker_tip.copy()
    adjusted_mount[0, 3] = .003
    _, adjusted, raw = resample_segment(records, marker_tip=adjusted_mount)
    np.testing.assert_allclose(np.asarray(raw[-1]["T_socket_usb_tip"])[0, 3], .003)
    np.testing.assert_allclose(adjusted[-1][:3, 3] - baseline[-1][:3, 3],
                               [.003, 0., 0.], atol=1e-12)


def test_resampling_rejects_implausible_marker_jump():
    port_mount, marker_tip = provisional_mounts()
    records = []
    for index, x in enumerate([0., 0., .1, .1]):
        socket_tip = _matrix(np.array([x, 0., -.02]), np.eye(3))
        port_usb = inverse(port_mount) @ socket_tip @ inverse(marker_tip)
        records.append({"frame_index": index, "timestamp_s": index * .05,
                        "quality": "ok", "T_port_usb_marker": port_usb.tolist()})
    with pytest.raises(ValueError, match="implausible"):
        resample_segment(records)


def test_lowpass_removes_tracking_jitter_but_keeps_slow_motion():
    rng = np.random.default_rng(0)
    t = np.arange(200) / 20
    slow = np.stack([.01 * np.sin(.5 * t), .002 * t, -.04 + .005 * t], axis=1)
    slow_angles = np.stack([.05 * np.sin(.3 * t), .02 * t, np.zeros_like(t)], axis=1)
    positions = slow + rng.normal(0, .0015, slow.shape)
    rotations = Rotation.from_rotvec(slow_angles + rng.normal(0, .01, slow.shape))
    filtered, filtered_rotations = lowpass_poses(positions, rotations, 2.0, 20)
    interior = slice(10, -10)
    assert (np.abs(filtered - slow)[interior].std() <
            .5 * np.abs(positions - slow)[interior].std())
    angle_error = (Rotation.from_rotvec(slow_angles).inv() * filtered_rotations).magnitude()
    raw_error = (Rotation.from_rotvec(slow_angles).inv() * rotations).magnitude()
    assert angle_error[interior].mean() < .5 * raw_error[interior].mean()


def test_mirror_reflects_lateral_offset_and_keeps_rotations_proper():
    port_mount, marker_tip = provisional_mounts()
    tilt = Rotation.from_rotvec([.05, .1, .2]).as_matrix()
    records = []
    for index, depth in enumerate([-.04, -.025, -.010, .008]):
        socket_tip = _matrix(np.array([.004, -.002, depth]), tilt)
        port_usb = inverse(port_mount) @ socket_tip @ inverse(marker_tip)
        records.append({"frame_index": index, "timestamp_s": index * .1,
                        "quality": "ok", "T_port_usb_marker": port_usb.tolist()})
    _, baseline, _ = resample_segment(records)
    _, mirrored, _ = resample_segment(records, mirror_socket_x=True)
    reflect = np.diag([-1., 1., 1., 1.])
    for original, flipped in zip(baseline, mirrored):
        np.testing.assert_allclose(flipped, reflect @ original @ reflect, atol=1e-9)
        assert np.linalg.det(flipped[:3, :3]) == pytest.approx(1.0)
    # Rotation about local X is unchanged; about Y and Z it reverses.
    np.testing.assert_allclose(Rotation.from_matrix(mirrored[-1][:3, :3]).as_rotvec(),
                               [.05, -.1, -.2], atol=1e-9)
