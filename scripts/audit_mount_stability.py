"""Flag possible within-clip changes in the two-marker rigid mount."""

import csv
import json
from pathlib import Path

import numpy as np

from mimicvla.vision.transforms import inverse


def main() -> None:
    root = Path("outputs/recalibration")
    rows = []
    for path in sorted(Path("outputs/dual_marker_original_mount/dual").glob(
            "*/tracking/trajectory.jsonl")):
        translations = []
        for line in path.read_text().splitlines():
            record = json.loads(line)
            markers = record["markers"]
            if "0" not in markers or "10" not in markers:
                continue
            marker0 = np.asarray(markers["0"]["T_camera_marker"])
            marker10 = np.asarray(markers["10"]["T_camera_marker"])
            translations.append((inverse(marker0) @ marker10)[:3, 3] * 1000)
        n = len(translations)
        row = {"clip": path.parent.parent.name, "co_visible_frames": n,
               "early_late_mount_translation_shift_mm": None,
               "possible_mount_change_or_pose_error": False}
        if n >= 24:
            third = n // 3
            translation = np.asarray(translations)
            early = np.median(translation[:third], axis=0)
            late = np.median(translation[-third:], axis=0)
            shift = float(np.linalg.norm(late - early))
            row.update(early_late_mount_translation_shift_mm=shift,
                       possible_mount_change_or_pose_error=shift > 4)
        rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    (root / "marker_mount_stability.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (root / "marker_mount_stability.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assessed = [row for row in rows if row["early_late_mount_translation_shift_mm"] is not None]
    print(f"Assessed {len(assessed)} clips; {sum(row['possible_mount_change_or_pose_error'] for row in assessed)} exceed 4 mm."
          " This is a tracking/mount review flag, not proof that a marker moved.")


if __name__ == "__main__":
    main()
