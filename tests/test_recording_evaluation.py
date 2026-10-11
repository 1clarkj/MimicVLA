"""Short marker gaps can be used without altering the raw track."""

import numpy as np

from mimicvla.evaluation.recordings import fill_short_gaps


def test_fill_short_gap_preserves_original_and_endpoints():
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = .008
    records = [
        {"quality": "ok", "timestamp_s": 0., "T_port_usb_marker": a.tolist()},
        {"quality": "missing_usb", "timestamp_s": .05, "T_port_usb_marker": None},
        {"quality": "ok", "timestamp_s": .1, "T_port_usb_marker": b.tolist()},
    ]
    filled, count = fill_short_gaps(records)
    assert count == 1
    assert records[1]["quality"] == "missing_usb"
    assert filled[1]["quality"] == "ok"
    assert filled[1]["quality_original"] == "missing_usb"
    np.testing.assert_allclose(np.asarray(filled[1]["T_port_usb_marker"])[:3, 3], [.004, 0, 0])


def test_reject_long_or_fast_gap():
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = .1
    records = [
        {"quality": "ok", "timestamp_s": 0., "T_port_usb_marker": a.tolist()},
        {"quality": "missing_usb", "timestamp_s": .05, "T_port_usb_marker": None},
        {"quality": "ok", "timestamp_s": .1, "T_port_usb_marker": b.tolist()},
    ]
    _, count = fill_short_gaps(records)
    assert count == 0
