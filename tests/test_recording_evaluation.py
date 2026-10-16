"""Short marker gaps can be used without altering the raw track."""

from pathlib import Path

import numpy as np
import pytest

from mimicvla.dataset.tracking.dual_recordings import resolve_batch
from mimicvla.dataset.tracking.recordings import _runs, fill_short_gaps, select_insertion_run


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


def test_pose_discontinuity_splits_replay_run():
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = .10
    records = [{"quality": "ok", "timestamp_s": i * .05,
                "T_port_usb_marker": (a if i < 3 else b).tolist(), "frame_index": i}
               for i in range(6)]
    assert [[r["frame_index"] for r in run] for run in _runs(records)] == [[0, 1, 2], [3, 4, 5]]


def test_insertion_after_long_tracking_loss_starts_new_clip():
    port = np.eye(4)
    port[:3, 3] = [0, 0, -0.100]
    records = [{"quality": "ok", "timestamp_s": i / 20,
                "T_port_usb_marker": port.tolist(), "frame_index": i}
               for i in range(5)]
    records += [{"quality": "missing_usb", "timestamp_s": i / 20,
                 "T_port_usb_marker": None, "frame_index": i}
                for i in range(5, 35)]
    # Construct relative marker poses whose socket tip is 1 to 20 mm deep.
    from mimicvla.vision.target_frames import provisional_mounts
    port_mount, tip_mount = provisional_mounts()
    for i in range(35, 65):
        depth = .001 + (i - 35) * .019 / 29
        target = np.eye(4)
        target[2, 3] = depth
        marker = np.linalg.inv(port_mount) @ target @ np.linalg.inv(tip_mount)
        records.append({"quality": "ok", "timestamp_s": i / 20,
                        "T_port_usb_marker": marker.tolist(), "frame_index": i})
    filled, count = fill_short_gaps(records)
    selected, details = select_insertion_run(filled)
    assert count == 0
    assert selected[0]["frame_index"] == 35
    assert details["approach_truncated"]


def test_recording_batches_keep_distinct_usb_mounts_and_outputs():
    old = resolve_batch(Path("video_data/double_marker"), None, None)
    new = resolve_batch(Path("video_data/double_marker_original_mount"), None, None)
    assert old == (-3.0, Path("outputs/dual_marker_comparison_tip_xminus3"))
    assert new == (0.0, Path("outputs/dual_marker_original_mount"))
    with pytest.raises(ValueError, match="Unknown recording batch"):
        resolve_batch(Path("video_data/mystery"), None, None)
    with pytest.raises(ValueError, match="separate --output-dir"):
        resolve_batch(Path("video_data/double_marker"), 0.0, None)
