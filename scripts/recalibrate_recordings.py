"""Estimate session-level marker-to-tip translations from recorded insertions.

The user confirmed that the physical plug enters the port in these videos.
Each clip contributes one robust interior estimate, so long clips cannot
dominate. Raw tracks are never modified. The result is a candidate calibration
with residuals, not an independent physical measurement of the tip geometry.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from mimicvla.vision.target_frames import marker_tip_with_x_offset, provisional_mounts


BATCHES = {
    "single_20mm": ("outputs/video_evaluation_pose_prior/20mm_*/tracking/trajectory.jsonl", 0.0),
    "single_12mm": ("outputs/video_evaluation_pose_prior/12mm_*/tracking/trajectory.jsonl", 0.0),
    "dual_original": ("outputs/dual_marker_original_mount/dual/*/tracking/trajectory.jsonl", 0.0),
    "dual_minus3": ("outputs/dual_marker_comparison/dual/*/tracking/trajectory.jsonl", -3.0),
}


def samples_for_track(path: Path, tip: np.ndarray) -> dict | None:
    port, _ = provisional_mounts()
    positions, jacobians = [], []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["quality"] != "ok":
            continue
        relative = np.asarray(row["T_port_usb_marker"], dtype=float)
        world_marker = port @ relative
        position = (world_marker @ tip)[:3, 3]
        if .008 <= position[2] <= .022:
            positions.append(position[:2] * 1000)
            jacobians.append(world_marker[:2, [0, 2]])
    if len(positions) < 8:
        return None
    return {"clip": path.parent.parent.name, "source_track": str(path),
            "interior_frames": len(positions),
            "xy_mm": np.median(positions, axis=0),
            "jacobian_xz": np.median(jacobians, axis=0)}


def _cost(samples: list[dict]) -> float:
    values = np.asarray([s["xy_mm"] for s in samples])
    centre = np.median(values, axis=0)
    return float(np.minimum(np.sum((values - centre) ** 2, axis=1), 16).sum())


def split_epochs(samples: list[dict], *, min_clips: int = 5,
                 penalty_mm2: float = 15) -> list[list[dict]]:
    """Split only when a sustained shift beats robust within-epoch variation."""
    count = len(samples)
    if count < 2 * min_clips:
        return [samples]
    values = np.asarray([s["xy_mm"] for s in samples])
    costs = np.full(count + 1, np.inf)
    previous = np.full(count + 1, -1, dtype=int)
    costs[0] = -penalty_mm2
    for end in range(min_clips, count + 1):
        for start in range(0, end - min_clips + 1):
            if not np.isfinite(costs[start]):
                continue
            if start:
                left = values[previous[start]:start]
                right = values[start:end]
                if np.linalg.norm(np.median(left, axis=0) -
                                  np.median(right, axis=0)) < 2:
                    continue
            candidate = costs[start] + _cost(samples[start:end]) + penalty_mm2
            if candidate < costs[end]:
                costs[end], previous[end] = candidate, start
    groups = []
    end = count
    while end:
        start = previous[end]
        groups.append(samples[start:end])
        end = start
    return groups[::-1]


def _fit_delta(positions: np.ndarray, jacobians: np.ndarray) -> np.ndarray:
    """Fit the two identifiable lateral translations with a weak size prior."""
    def residual(delta):
        prediction = positions + np.einsum("nij,j->ni", jacobians, delta)
        return np.r_[prediction.ravel(), delta / 8]
    return least_squares(residual, np.zeros(2), loss="soft_l1", f_scale=1.5,
                         bounds=([-12, -8], [12, 8])).x


def fit_epoch(samples: list[dict], base_tip: np.ndarray) -> dict:
    positions = np.asarray([s["xy_mm"] for s in samples])
    jacobians = np.asarray([s["jacobian_xz"] for s in samples])
    # Only X and Z are inferred; the physical 30 mm marker-to-tip Y stays fixed.
    # A weak 8 mm prior prevents a tiny epoch from inventing a large correction.
    delta = _fit_delta(positions, jacobians)
    corrected = positions + np.einsum("nij,j->ni", jacobians, delta)
    held_out = []
    if len(samples) >= 5:
        for index in range(len(samples)):
            keep = np.arange(len(samples)) != index
            loo_delta = _fit_delta(positions[keep], jacobians[keep])
            held_out.append(np.linalg.norm(positions[index] +
                                           jacobians[index] @ loo_delta))
    tip = base_tip.copy()
    tip[0, 3] += delta[0] / 1000
    tip[2, 3] += delta[1] / 1000
    return {"clip_count": len(samples), "first_clip": samples[0]["clip"],
            "last_clip": samples[-1]["clip"],
            "clips": [s["clip"] for s in samples],
            "delta_marker_xz_mm": delta.tolist(),
            "T_marker0_usb_tip": tip.tolist(),
            "raw_median_xy_mm": np.median(positions, axis=0).tolist(),
            "corrected_median_xy_mm": np.median(corrected, axis=0).tolist(),
            "raw_radial_mm_p50": float(np.median(np.linalg.norm(positions, axis=1))),
            "corrected_radial_mm_p50": float(np.median(np.linalg.norm(corrected, axis=1))),
            "corrected_radial_mm_p90": float(np.percentile(np.linalg.norm(corrected, axis=1), 90)),
            "leave_one_clip_out_radial_mm_p50": float(np.median(held_out)) if held_out else None,
            "leave_one_clip_out_radial_mm_p90": float(np.percentile(held_out, 90)) if held_out else None,
            "within_full_clearance_clips": int(np.sum(np.all(np.abs(corrected) <= .7, axis=1)))}


def main() -> None:
    output = Path("outputs/recalibration")
    output.mkdir(parents=True, exist_ok=True)
    report = {"method": "robust per-clip median at tracked depth 8-22 mm; session change points; fit marker X/Z translation only",
              "caveat": "Physical insertion is user-confirmed, but tracking and human correction leave residual error; this is a candidate task calibration, not ground truth.",
              "batches": {}}
    rows = []
    for name, (pattern, base_x_mm) in BATCHES.items():
        base = marker_tip_with_x_offset(base_x_mm)
        paths = sorted(Path().glob(pattern))
        samples = [sample for path in sorted(Path().glob(pattern))
                   if (sample := samples_for_track(path, base)) is not None]
        if name == "single_12mm" and len(samples) < 5:
            source = report["batches"]["single_20mm"]["epochs"][0]
            epochs = [{**source, "first_clip": paths[0].parent.parent.name,
                       "last_clip": paths[-1].parent.parent.name,
                       "clips": [path.parent.parent.name for path in paths],
                       "clip_count": 0,
                       "raw_median_xy_mm": None,
                       "corrected_median_xy_mm": None,
                       "raw_radial_mm_p50": None,
                       "corrected_radial_mm_p50": None,
                       "corrected_radial_mm_p90": None,
                       "leave_one_clip_out_radial_mm_p50": None,
                       "leave_one_clip_out_radial_mm_p90": None,
                       "within_full_clearance_clips": None,
                       "borrowed_from": "single_20mm first epoch; too few 12mm clips to fit"}]
        else:
            epochs = [fit_epoch(group, base) for group in split_epochs(samples)] if samples else []
        report["batches"][name] = {"eligible_calibration_clips": len(samples),
                                     "epochs": epochs}
        for epoch_index, epoch in enumerate(epochs):
            for sample in samples:
                if sample["clip"] not in epoch["clips"]:
                    continue
                delta = np.asarray(epoch["delta_marker_xz_mm"])
                adjusted = sample["xy_mm"] + sample["jacobian_xz"] @ delta
                rows.append({"batch": name, "epoch": epoch_index,
                             "clip": sample["clip"],
                             "interior_frames": sample["interior_frames"],
                             "raw_x_mm": sample["xy_mm"][0],
                             "raw_y_mm": sample["xy_mm"][1],
                             "corrected_x_mm": adjusted[0],
                             "corrected_y_mm": adjusted[1]})
    (output / "candidate_mounts.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "clip_residuals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="row", sharey="col")
    for index, name in enumerate(("single_20mm", "dual_original")):
        batch_rows = [row for row in rows if row["batch"] == name]
        for axis_index, dimension in enumerate(("x", "y")):
            axis = axes[index, axis_index]
            axis.plot([row[f"raw_{dimension}_mm"] for row in batch_rows], ".-",
                      ms=4, lw=.8, label="original")
            axis.plot([row[f"corrected_{dimension}_mm"] for row in batch_rows], ".-",
                      ms=4, lw=.8, label="recalibrated")
            for epoch_index in range(1, len(report["batches"][name]["epochs"])):
                boundary = next(i for i, row in enumerate(batch_rows)
                                if row["epoch"] == epoch_index)
                axis.axvline(boundary - .5, color="gray", alpha=.5, linestyle="--")
            axis.axhline(0, color="black", lw=.7)
            axis.axhspan(-.7, .7, color="green", alpha=.1)
            axis.set(title=f"{name}: socket {dimension.upper()} offset",
                     ylabel="mm", xlabel="recording order")
            axis.grid(alpha=.2)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output / "calibration_drift.png", dpi=150)
    plt.close(fig)
    for name, batch in report["batches"].items():
        print(name, batch["eligible_calibration_clips"], "clips", len(batch["epochs"]), "epochs")
        for epoch in batch["epochs"]:
            print(" ", epoch["first_clip"], epoch["last_clip"],
                  "delta x/z mm", np.round(epoch["delta_marker_xz_mm"], 2),
                  "residual p90 mm", (None if epoch["corrected_radial_mm_p90"] is None
                                      else round(epoch["corrected_radial_mm_p90"], 2)))


if __name__ == "__main__":
    main()
