"""Gap filling, run splitting and insertion-segment selection for tracked recordings."""

import json

import numpy as np

from mimicvla.dataset.build import _runs, select_insertion_run
from mimicvla.vision.track import fill_short_gaps


def test_fill_short_gap_keeps_original_label_and_interpolates():
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = .008
    records = [
        {"quality": "ok", "timestamp_s": 0., "T_port_usb_marker": a.tolist()},
        {"quality": "missing_usb", "timestamp_s": .05, "T_port_usb_marker": None},
        {"quality": "ok", "timestamp_s": .1, "T_port_usb_marker": b.tolist()},
    ]
    assert fill_short_gaps(records) == 1
    assert records[1]["quality"] == "ok"
    assert records[1]["quality_original"] == "missing_usb"
    assert records[1]["pose_source"] == "interpolated"
    np.testing.assert_allclose(np.asarray(records[1]["T_port_usb_marker"])[:3, 3], [.004, 0, 0])


def test_reject_fast_gap():
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = .1
    records = [
        {"quality": "ok", "timestamp_s": 0., "T_port_usb_marker": a.tolist()},
        {"quality": "missing_usb", "timestamp_s": .05, "T_port_usb_marker": None},
        {"quality": "ok", "timestamp_s": .1, "T_port_usb_marker": b.tolist()},
    ]
    assert fill_short_gaps(records) == 0
    assert records[1]["quality"] == "missing_usb"


def test_pose_discontinuity_splits_replay_run():
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = .10
    records = [{"quality": "ok", "timestamp_s": i * .05,
                "T_port_usb_marker": (a if i < 3 else b).tolist(), "frame_index": i}
               for i in range(6)]
    assert [[r["frame_index"] for r in run] for run in _runs(records)] == [[0, 1, 2], [3, 4, 5]]


def test_insertion_after_long_tracking_loss_is_selected():
    def record(i, depth):
        tip = np.eye(4)
        tip[2, 3] = depth
        return {"quality": "ok", "timestamp_s": i / 20, "frame_index": i,
                "T_port_usb_marker": tip.tolist(), "T_socket_usb_tip": tip.tolist()}
    records = [record(i, -.1) for i in range(5)]
    records += [{"quality": "missing_usb", "timestamp_s": i / 20, "frame_index": i,
                 "T_port_usb_marker": None, "T_socket_usb_tip": None} for i in range(5, 35)]
    records += [record(i, .001 + (i - 35) * .019 / 29) for i in range(35, 65)]
    selected, details = select_insertion_run(records)
    assert selected[0]["frame_index"] == 35
    assert details["peak_depth_m"] > .019
    none, reason = select_insertion_run(records[:5])
    assert none is None and "reason" in reason


def test_build_dataset_from_a_straight_insertion(tmp_path):
    from mimicvla.dataset.build import build

    tracking = tmp_path / "tracking"
    tracking.mkdir()
    lines = []
    # Approach, pause at the mouth, then push in slowly, as the human clips do.
    depths = np.r_[np.linspace(-.04, -.004, 20), np.full(15, -.004), np.linspace(-.004, .018, 30)]
    seated = np.array([[1., .02, .019, -.001], [-.021, .998, .056, -.001],
                       [-.017, -.056, .998, .015], [0, 0, 0, 1]])  # from a recorded clip
    seated[:3, :3] = np.linalg.svd(seated[:3, :3])[0] @ np.linalg.svd(seated[:3, :3])[2]
    for i, depth in enumerate(depths):
        tip = seated.copy()
        tip[2, 3] = depth
        lines.append({"frame_index": i, "timestamp_s": i / 15, "quality": "ok",
                      "T_port_usb_marker": tip.tolist(), "T_socket_usb_tip": tip.tolist()})
    (tracking / "straight.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    (tracking / "short.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines[:10]))
    result = build(tracking, tmp_path / "dataset", videos=False)
    assert result["episodes"] == 1 and result["rejected"] == 1
    manifest = json.loads((tmp_path / "dataset" / "capture_manifest.json").read_text())
    episode = manifest["episodes_manifest"][0]
    assert episode["clip"] == "straight" and episode["final_depth_m"] >= .012
    assert episode["frames"] == episode["last_source_step"] - episode["first_source_step"] + 1
