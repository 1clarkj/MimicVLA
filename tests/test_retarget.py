"""Checks that marker-frame measurements become feasible controller targets."""

import numpy as np

from mimicvla.retarget import _densify, _matrix, resample_segment, valid_segment
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
