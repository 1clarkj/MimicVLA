"""Session calibration detects sustained changes without per-clip offsets."""

import numpy as np

from mimicvla.dataset.tracking import recalibrate as module


def test_sustained_mount_shift_creates_two_transforms():
    samples = [{"clip": f"clip_{i:02d}", "xy_mm": np.array([x, .5]),
                "jacobian_xz": np.array([[1., 0.], [0., -1.]])}
               for i, x in enumerate([4., 4.1, 3.9, 4.2, 4.,
                                       8., 7.9, 8.1, 8., 8.2])]
    groups = module.split_epochs(samples)
    assert [len(group) for group in groups] == [5, 5]
    tip = module.marker_tip_with_x_offset(0)
    first, second = [module.fit_epoch(group, tip) for group in groups]
    assert abs(first["delta_marker_xz_mm"][0] + 4) < .3
    assert abs(second["delta_marker_xz_mm"][0] + 8) < .3
    assert first["leave_one_clip_out_radial_mm_p90"] < 1
    assert second["leave_one_clip_out_radial_mm_p90"] < 1
