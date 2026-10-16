import json

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

from mimicvla.vision.calibrate import calibrate, load_calibration, make_board
from mimicvla.vision.track import _select_pose, detect_pair, track_video
from mimicvla.vision.transforms import inverse, pose_matrix
from mimicvla.vision.target_frames import marker_tip_with_x_offset, provisional_mounts, socket_from_tip


def test_transform_inverse_and_composition():
    first = pose_matrix([0.1, -0.2, 0.3], [0.4, 0.5, -0.2])
    second = pose_matrix([-0.2, 0.3, 0.1], [0.01, -0.03, 0.1])
    np.testing.assert_allclose(first @ inverse(first), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(inverse(first @ second), inverse(second) @ inverse(first), atol=1e-12)


def test_ippe_pose_selection_prefers_continuous_normal():
    previous = np.eye(4)
    flipped = np.eye(4)
    flipped[:3, :3] = Rotation.from_rotvec([0, 1.84, 0]).as_matrix()
    smooth = np.eye(4)
    smooth[:3, :3] = Rotation.from_rotvec([0, 0.016, 0]).as_matrix()
    error, pose, method = _select_pose([(0.27, flipped), (1.17, smooth)], previous)
    assert error == 1.17
    assert method == "temporal_continuity"
    np.testing.assert_allclose(pose, smooth)
    error, pose, method = _select_pose([(0.27, flipped), (2.0, smooth)], previous)
    assert error == 0.27
    assert method == "lowest_reprojection"
    np.testing.assert_allclose(pose, flipped)
    # An unambiguous near-insertion orientation can recover after a long
    # occlusion even when the previous branch is no longer trustworthy.
    error, pose, method = _select_pose([(0.27, flipped), (1.17, smooth)],
                                       flipped, expected=previous)
    assert error == 1.17
    assert method == "task_orientation_prior"
    np.testing.assert_allclose(pose, smooth)


def test_measured_marker_offsets_and_relative_targets():
    port_mount, marker_tip = provisional_mounts()
    np.testing.assert_allclose(port_mount[:3, 3], [.016, -.045, 0])
    np.testing.assert_allclose(marker_tip[:3, 3], [0, .030, -.003])
    np.testing.assert_allclose(inverse(port_mount)[:3, 3], [-.016, -.045, 0])
    np.testing.assert_allclose(marker_tip[:3, :3] @ [0, 1, 0], [0, 0, -1])
    np.testing.assert_allclose(marker_tip[:3, :3] @ [0, 0, 1], [0, 1, 0])
    for mount in (port_mount, marker_tip):
        np.testing.assert_allclose(mount[:3, :3].T @ mount[:3, :3], np.eye(3))
        np.testing.assert_allclose(np.linalg.det(mount[:3, :3]), 1)
    # A marker pair constructed from an aligned tip must recover identity.
    relative_markers = inverse(port_mount) @ inverse(marker_tip)
    record = {"quality": "ok", "T_port_usb_marker": relative_markers.tolist()}
    np.testing.assert_allclose(socket_from_tip(record, port_mount, marker_tip), np.eye(4), atol=1e-12)


def test_new_mount_marker_center_is_plus_3mm_from_tip_in_x():
    marker_tip = marker_tip_with_x_offset(-3)
    np.testing.assert_allclose(marker_tip[:3, 3], [-.003, .030, -.003])
    # X is shared by the marker and tip frames, so inversion gives the
    # reported marker-centre position relative to the tip centre.
    np.testing.assert_allclose(inverse(marker_tip)[0, 3], .003)


def test_charuco_calibration_from_synthetic_views(tmp_path):
    board_image = make_board().generateImage((600, 840))
    source = np.float32([[0, 0], [599, 0], [599, 839], [0, 839]])
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    for index in range(8):
        shift = (index - 4) * 15
        target = np.float32([[350 + shift, 70 + index * 6], [850 + shift, 90 - index * 5],
                             [800 - shift, 780 - index * 10], [360 - shift, 750 + index * 8]])
        image = cv2.warpPerspective(board_image, cv2.getPerspectiveTransform(source, target),
                                    (1280, 960), borderValue=255)
        cv2.imwrite(str(photo_dir / f"view_{index:02d}.png"), image)
    output = tmp_path / "calibration.json"
    result = calibrate(photo_dir, output, stride=1)
    assert len(result["accepted_views"]) >= 6
    assert result["board"]["ids"] == list(range(10, 27))
    assert np.isfinite(result["rms_reprojection_px"])
    assert load_calibration(output)["image_size_wh"] == [1280, 960]


def test_tracking_synthetic_video_with_missing_marker(tmp_path):
    size = (640, 480)
    camera = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "schema": "mimicvla.camera_calibration.v1", "image_size_wh": list(size),
        "camera_matrix": camera.tolist(), "distortion_coefficients": [0, 0, 0, 0, 0],
    }))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary)
    port = cv2.aruco.generateImageMarker(dictionary, 1, 150)
    usb = cv2.aruco.generateImageMarker(dictionary, 0, 60)
    video = tmp_path / "demo.mp4"
    with imageio.get_writer(video, fps=10, codec="libx264", quality=9) as writer:
        for index in range(4):
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)
            frame[150:300, 100:250] = cv2.cvtColor(port, cv2.COLOR_GRAY2RGB)
            if index != 2:
                frame[180:240, 380 + index * 5:440 + index * 5] = cv2.cvtColor(
                    usb, cv2.COLOR_GRAY2RGB)
            writer.append_data(frame)
    sample = np.full((480, 640, 3), 255, dtype=np.uint8)
    sample[150:300, 100:250] = cv2.cvtColor(port, cv2.COLOR_GRAY2BGR)
    sample[180:240, 380:440] = cv2.cvtColor(usb, cv2.COLOR_GRAY2BGR)
    record, _ = detect_pair(sample, detector, camera, np.zeros(5), .020, .050)
    assert record["quality"] == "ok"
    compact_record, preview = detect_pair(sample, detector, camera, np.zeros(5),
                                          .020, .050, annotate=False)
    assert compact_record["quality"] == "ok"
    assert preview is None
    relative = np.asarray(record["T_port_usb_marker"])
    np.testing.assert_allclose(relative[:3, :3] @ relative[:3, :3].T, np.eye(3), atol=1e-6)
    assert relative[0, 3] > 0  # USB marker appears to the port marker's right.
    output = tmp_path / "tracked"
    metadata = track_video(str(video), calibration, output, usb_marker_m=.020)
    records = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
    assert metadata["frames"] == 4
    assert metadata["valid_relative_poses"] == 3
    assert records[2]["quality"] == "missing_usb"
    assert records[2]["T_port_usb_marker"] is None
    assert all(records[i]["timestamp_s"] < records[i + 1]["timestamp_s"] for i in range(3))
    assert (output / "trajectory.png").exists()
    assert len(list((output / "frames").glob("*.jpg"))) == 4
