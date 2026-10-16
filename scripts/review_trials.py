#!/usr/bin/env python3
"""Local, persistent review queue for dual- and single-marker insertions."""

import argparse
import csv
import json
import re
import subprocess
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("stem", "decision", "human_success", "tracking", "replay", "notes",
          "override", "override_reasons", "updated_at")
CHOICES = {
    "decision": {"unreviewed", "accept", "hold", "reject"},
    "human_success": {"unknown", "yes", "no", "unclear"},
    "tracking": {"unknown", "good", "questionable", "bad"},
    "replay": {"unknown", "good", "questionable", "bad"},
}


def load_trials(results_dir: Path, scores_file: Path) -> list[dict]:
    rows = json.loads((results_dir / "results.json").read_text())
    scores = json.loads(scores_file.read_text())
    recalibrated_path = ROOT / "outputs/recalibration/replay_results.json"
    recalibrated = ({row["clip"]: row for row in json.loads(recalibrated_path.read_text())}
                    if recalibrated_path.exists() else {})
    fallback_path = ROOT / "outputs/recalibration/fallback_results.json"
    fallbacks = ({row["clip"]: row for row in json.loads(fallback_path.read_text())}
                 if fallback_path.exists() else {})
    stability_path = ROOT / "outputs/recalibration/marker_mount_stability.json"
    stability = ({row["clip"]: row for row in json.loads(stability_path.read_text())}
                 if stability_path.exists() else {})
    contact_path = ROOT / "outputs/recalibration/guided_contact_audit.json"
    contact = ({(row["clip"], row["tracking_mode"]): row
                for row in json.loads(contact_path.read_text())}
               if contact_path.exists() else {})
    no_funnel_queue_path = ROOT / "outputs/recalibration/wide_no_funnel_review_queue.csv"
    if no_funnel_queue_path.exists():
        with no_funnel_queue_path.open(newline="") as stream:
            no_funnel_queue = {item["clip"]: item for item in csv.DictReader(stream)}
    else:
        no_funnel_queue = {}
    additional_queue_path = ROOT / "outputs/recalibration/additional_wide_no_funnel_review_queue.csv"
    if additional_queue_path.exists():
        with additional_queue_path.open(newline="") as stream:
            no_funnel_queue.update({item["clip"]: item for item in csv.DictReader(stream)})
    by_stem = {}
    for row in scores:
        if row["batch"] == "dual_marker_original_mount" and row["mode"] == "dual":
            by_stem[Path(row["video"]).stem] = row
    marker0 = {Path(row["video"]).stem: row for row in rows if row["mode"] == "marker0_only"}
    trials = []
    for row in rows:
        if row["mode"] != "dual":
            continue
        stem = Path(row["video"]).stem
        score = by_stem.get(stem)
        updated = recalibrated.get(stem, {})
        fallback = fallbacks.get(stem, {})
        mount_stability = stability.get(stem, {})
        guided_contact = contact.get((stem, "primary"), {})
        fallback_contact = contact.get((stem, "marker0_fallback"), {})
        no_funnel = no_funnel_queue.get(stem, {})
        if score is None:
            raise ValueError(f"Missing current 12 mm score for {stem}")
        source = (ROOT / row["video"]).resolve()
        trial_dir = results_dir / "dual" / stem
        historical_replay = trial_dir / "replay/replay.mp4"
        guided_replay = ROOT / "outputs/recalibration/replays/dual_original" / stem / "training/replay.mp4"
        trial = {
            "stem": stem,
            "cohort": "later" if stem >= "10112026_222522" else "pilot",
            "source": str(source),
            "plot": str(trial_dir / "tracking" / "trajectory.png"),
            "replay_video": str(historical_replay if historical_replay.exists() else guided_replay),
            "guided_replay_video": str(guided_replay),
            "fallback_replay_video": str(ROOT / "outputs/recalibration/replays/dual_original" /
                                         stem / "marker0_fallback/training/replay.mp4"),
            "no_funnel_replay_video": str(ROOT / no_funnel["replay_video"])
            if no_funnel else "",
            "no_funnel_review_tier": no_funnel.get("tier", ""),
            "no_funnel_concerns": no_funnel.get("concerns", ""),
            "no_funnel_prefix_max_wall_force_n": no_funnel.get(
                "peak_contact_n_to_first_entry"),
            "frames": row["frames"],
            "valid_fraction": row["valid_fraction"],
            "gap": row["longest_raw_tracking_gap_frames"],
            "interpolated": row["interpolated_frames"],
            "implausible_steps": row["implausible_pose_steps"],
            "agreement_p50_mm": row.get("marker_agreement_mm_p50"),
            "agreement_p95_mm": row.get("marker_agreement_mm_p95"),
            "disagreement_over_5mm_frames": row["marker_disagreement_over_5mm_frames"],
            "eligible": (updated.get("status") == "replayable" if updated else bool(row["eligible"])),
            "reason": row.get("reason", ""),
            "segment_frames": row.get("segment_frames"),
            "start_frame": row.get("start_frame"),
            "end_frame": row.get("end_frame"),
            "depth_mm": score.get("max_axial_depth_mm"),
            "success": bool(score["episode_success"]),
            "quality_ok": bool(score.get("replay_quality_ok", False)),
            "bad_collision_steps": score.get("bad_collision_steps"),
            "wall_contact_steps": score.get("wall_contact_steps"),
            "wall_force_n": row.get("max_wall_normal_force_n"),
            "wall_penetration_mm": row.get("max_wall_penetration_mm"),
            "marker0_eligible": bool(marker0.get(stem, {}).get("eligible", False)),
            "calibrated_strict_success": updated.get("strict_success"),
            "calibrated_training_success": updated.get("training_success"),
            "calibrated_training_candidate": updated.get("training_prefix_candidate"),
            "calibrated_training_depth_mm": updated.get("training_max_depth_mm"),
            "calibrated_strict_depth_mm": updated.get("strict_max_depth_mm"),
            "calibration_approach_truncated": updated.get("approach_truncated"),
            "marker0_guided_fallback_candidate": fallback.get("training_prefix_candidate", False),
            "within_clip_mount_shift_mm": mount_stability.get(
                "early_late_mount_translation_shift_mm"),
            "guided_prefix_max_wall_force_n": guided_contact.get(
                "max_wall_force_n_before_success"),
            "fallback_prefix_max_wall_force_n": fallback_contact.get(
                "max_wall_force_n_before_success"),
        }
        trial["priority"] = (0 if trial["eligible"] and (trial["quality_ok"] or
                                                          trial["calibrated_training_candidate"] or
                                                          trial["marker0_guided_fallback_candidate"]) else
                             1 if trial["eligible"] else 2)
        trials.append(trial)
    single_results = ROOT / "outputs/video_evaluation_pose_prior/results.json"
    if single_results.exists():
        single_scores = {f"{Path(row['video']).parent.name.split('mm_')[0]}mm_{Path(row['video']).stem}": row
                         for row in scores if row["batch"] == "single_marker_original"
                         and row["mode"] == "single_marker"}
        for row in json.loads(single_results.read_text()):
            stem = f"{row['usb_marker_mm']}mm_{Path(row['video']).stem}"
            score = single_scores.get(stem, {})
            updated = recalibrated.get(stem, {})
            batch = f"single_{row['usb_marker_mm']}mm"
            source = (ROOT / row["video"]).resolve()
            trial_dir = ROOT / "outputs/video_evaluation_pose_prior" / stem
            guided = ROOT / "outputs/recalibration/replays" / batch / stem / "training/replay.mp4"
            historical_replay = trial_dir / "replay/replay.mp4"
            guided_contact = contact.get((stem, "primary"), {})
            no_funnel = no_funnel_queue.get(stem, {})
            trial = {
                "stem": stem, "cohort": batch,
                "source": str(source),
                "plot": str(trial_dir / "tracking/trajectory.png"),
                "replay_video": str(historical_replay if historical_replay.exists() else guided),
                "guided_replay_video": str(guided),
                "fallback_replay_video": "",
                "no_funnel_replay_video": str(ROOT / no_funnel["replay_video"])
                if no_funnel else "",
                "no_funnel_review_tier": no_funnel.get("tier", ""),
                "no_funnel_concerns": no_funnel.get("concerns", ""),
                "no_funnel_prefix_max_wall_force_n": no_funnel.get(
                    "peak_contact_n_to_first_entry"),
                "frames": row["frames"], "valid_fraction": row["valid_fraction"],
                "gap": row["longest_raw_tracking_gap_frames"],
                "interpolated": row["interpolated_frames"],
                "implausible_steps": row["implausible_pose_steps"],
                "agreement_p50_mm": None, "agreement_p95_mm": None,
                "disagreement_over_5mm_frames": 0,
                "eligible": (updated.get("status") == "replayable" if updated else bool(row["eligible"])),
                "reason": row.get("reason", ""),
                "segment_frames": row.get("segment_frames"),
                "start_frame": updated.get("start_frame", row.get("start_frame")),
                "end_frame": updated.get("end_frame", row.get("end_frame")),
                "depth_mm": score.get("max_axial_depth_mm"),
                "success": bool(score.get("episode_success", row.get("success"))),
                "quality_ok": bool(score.get("replay_quality_ok", row.get("replay_quality_ok"))),
                "bad_collision_steps": score.get("bad_collision_steps"),
                "wall_contact_steps": score.get("wall_contact_steps"),
                "wall_force_n": row.get("max_wall_normal_force_n"),
                "wall_penetration_mm": (row.get("max_wall_penetration_m", 0) * 1000
                                        if row.get("max_wall_penetration_m") is not None else None),
                "marker0_eligible": False,
                "calibrated_strict_success": updated.get("strict_success"),
                "calibrated_training_success": updated.get("training_success"),
                "calibrated_training_candidate": updated.get("training_prefix_candidate"),
                "calibrated_training_depth_mm": updated.get("training_max_depth_mm"),
                "calibrated_strict_depth_mm": updated.get("strict_max_depth_mm"),
                "calibration_approach_truncated": updated.get("approach_truncated"),
                "marker0_guided_fallback_candidate": False,
                "within_clip_mount_shift_mm": None,
                "guided_prefix_max_wall_force_n": guided_contact.get(
                    "max_wall_force_n_before_success"),
                "fallback_prefix_max_wall_force_n": None,
            }
            trial["priority"] = (0 if trial["eligible"] and
                                 (trial["quality_ok"] or trial["calibrated_training_candidate"])
                                 else 1 if trial["eligible"] else 2)
            trials.append(trial)
    older_results = ROOT / "outputs/dual_marker_comparison_tip_xminus3/results.json"
    if older_results.exists():
        older_rows = json.loads(older_results.read_text())
        older_scores = {Path(row["video"]).stem: row for row in scores
                        if row["batch"] == "dual_marker_offset_minus3mm"
                        and row["mode"] == "dual"}
        marker0_rows = {Path(row["video"]).stem: row for row in older_rows
                        if row["mode"] == "marker0_only"}
        for row in older_rows:
            if row["mode"] != "dual":
                continue
            stem = Path(row["video"]).stem
            score = older_scores.get(stem, {})
            updated = recalibrated.get(stem, {})
            source = (ROOT / row["video"]).resolve()
            trial_dir = ROOT / "outputs/dual_marker_comparison_tip_xminus3/dual" / stem
            guided = ROOT / "outputs/recalibration/replays/dual_minus3" / stem / "training/replay.mp4"
            historical = trial_dir / "replay/replay.mp4"
            guided_contact = contact.get((stem, "primary"), {})
            no_funnel = no_funnel_queue.get(stem, {})
            trial = {
                "stem": stem, "cohort": "dual_minus3", "source": str(source),
                "plot": str(ROOT / "outputs/dual_marker_comparison/dual" / stem /
                            "tracking/trajectory.png"),
                "replay_video": str(historical if historical.exists() else guided),
                "guided_replay_video": str(guided), "fallback_replay_video": "",
                "no_funnel_replay_video": str(ROOT / no_funnel["replay_video"])
                if no_funnel else "",
                "no_funnel_review_tier": no_funnel.get("tier", ""),
                "no_funnel_concerns": no_funnel.get("concerns", ""),
                "no_funnel_prefix_max_wall_force_n": no_funnel.get(
                    "peak_contact_n_to_first_entry"),
                "frames": row["frames"], "valid_fraction": row["valid_fraction"],
                "gap": row["longest_raw_tracking_gap_frames"],
                "interpolated": row["interpolated_frames"],
                "implausible_steps": row["implausible_pose_steps"],
                "agreement_p50_mm": row.get("marker_agreement_mm_p50"),
                "agreement_p95_mm": row.get("marker_agreement_mm_p95"),
                "disagreement_over_5mm_frames": row.get("marker_disagreement_over_5mm_frames", 0),
                "eligible": (updated.get("status") == "replayable" if updated else bool(row["eligible"])),
                "reason": row.get("reason", ""),
                "segment_frames": updated.get("segment_frames", row.get("segment_frames")),
                "start_frame": updated.get("start_frame", row.get("start_frame")),
                "end_frame": updated.get("end_frame", row.get("end_frame")),
                "depth_mm": score.get("max_axial_depth_mm"),
                "success": bool(score.get("episode_success", row.get("success"))),
                "quality_ok": bool(score.get("replay_quality_ok", row.get("replay_quality_ok"))),
                "bad_collision_steps": score.get("bad_collision_steps", row.get("bad_collision_steps")),
                "wall_contact_steps": score.get("wall_contact_steps", row.get("wall_contact_steps")),
                "wall_force_n": row.get("max_wall_normal_force_n"),
                "wall_penetration_mm": row.get("max_wall_penetration_mm"),
                "marker0_eligible": bool(marker0_rows.get(stem, {}).get("eligible", False)),
                "calibrated_strict_success": updated.get("strict_success"),
                "calibrated_training_success": updated.get("training_success"),
                "calibrated_training_candidate": updated.get("training_prefix_candidate"),
                "calibrated_training_depth_mm": updated.get("training_max_depth_mm"),
                "calibrated_strict_depth_mm": updated.get("strict_max_depth_mm"),
                "calibration_approach_truncated": updated.get("approach_truncated"),
                "marker0_guided_fallback_candidate": False,
                "within_clip_mount_shift_mm": None,
                "guided_prefix_max_wall_force_n": guided_contact.get(
                    "max_wall_force_n_before_success"),
                "fallback_prefix_max_wall_force_n": None,
            }
            trial["priority"] = (0 if trial["eligible"] and
                                 (trial["quality_ok"] or trial["calibrated_training_candidate"])
                                 else 1 if trial["eligible"] else 2)
            trials.append(trial)
    return sorted(trials, key=lambda trial: (trial["priority"], trial["stem"]))


def read_decisions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        decisions = {}
        for row in csv.DictReader(stream):
            row.setdefault("override", "False")
            row.setdefault("override_reasons", "")
            decisions[row["stem"]] = row
        return decisions


def validate_decision(payload: dict, trial: dict) -> dict:
    row = {field: str(payload.get(field, "")).strip() for field in
           ("stem", "decision", "human_success", "tracking", "replay", "notes")}
    for field, allowed in CHOICES.items():
        if row[field] not in allowed:
            raise ValueError(f"Invalid {field}: {row[field]}")
    if row["stem"] != trial["stem"]:
        raise ValueError("Trial ID mismatch")
    if len(row["notes"]) > 2000:
        raise ValueError("Notes must be at most 2000 characters")
    requested_override = payload.get("override", False)
    if not isinstance(requested_override, bool):
        raise ValueError("Override must be true or false")
    failed_checks = []
    if row["decision"] == "accept":
        if not trial["eligible"]:
            failed_checks.append("no_replayable_span")
        if row["human_success"] != "yes":
            failed_checks.append("human_insertion_" + row["human_success"])
        if row["tracking"] != "good":
            failed_checks.append("tracking_" + row["tracking"])
        if row["replay"] != "good":
            failed_checks.append("replay_" + row["replay"])
        if failed_checks and not requested_override:
            raise ValueError("Check Manual override to accept despite: " + ", ".join(failed_checks))
        if failed_checks and not row["notes"]:
            raise ValueError("Add a reason in Notes for a manual override")
    row["override"] = "True" if failed_checks else "False"
    row["override_reasons"] = "|".join(failed_checks)
    row["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return row


def write_decisions(path: Path, decisions: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", dir=path.parent, delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(decisions[stem] for stem in sorted(decisions))
        temporary = Path(stream.name)
    temporary.replace(path)


def handler_factory(trials: list[dict], decisions_path: Path, preview_dir: Path):
    by_stem = {trial["stem"]: trial for trial in trials}
    lock = threading.Lock()
    preview_lock = threading.Lock()
    html = (ROOT / "scripts" / "trial_review.html").read_bytes()

    class ReviewHandler(BaseHTTPRequestHandler):
        def send_bytes(self, data: bytes, content_type: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, value, code: int = 200):
            self.send_bytes(json.dumps(value).encode(), "application/json; charset=utf-8", code)

        def send_file(self, path: Path, content_type: str):
            if not path.is_file():
                self.send_error(404, "Evidence file missing")
                return
            size = path.stat().st_size
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
            start = int(match.group(1)) if match else 0
            end = min(int(match.group(2)), size - 1) if match and match.group(2) else size - 1
            if start >= size or end < start:
                self.send_error(416, "Invalid byte range")
                return
            self.send_response(206 if match else 200)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if match:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send_bytes(html, "text/html; charset=utf-8")
            elif path == "/api/trials":
                self.send_json({"trials": trials, "decisions": read_decisions(decisions_path)})
            elif path == "/api/decisions.csv":
                if decisions_path.exists():
                    self.send_file(decisions_path, "text/csv; charset=utf-8")
                else:
                    self.send_bytes((",".join(FIELDS) + "\n").encode(), "text/csv; charset=utf-8")
            elif path.startswith("/media/"):
                parts = path.split("/")
                if len(parts) != 4 or parts[2] not in {"source", "replay", "guided", "no_funnel", "fallback", "plot"}:
                    self.send_error(404)
                    return
                kind, stem = parts[2], unquote(parts[3])
                trial = by_stem.get(stem)
                if trial is None:
                    self.send_error(404)
                    return
                if kind == "source":
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    preview = preview_dir / f"{stem}.mp4"
                    source = Path(trial["source"])
                    if not source.is_file():
                        self.send_error(404, "Original source clip missing")
                        return
                    with preview_lock:
                        if (not preview.exists() or
                                preview.stat().st_mtime_ns < source.stat().st_mtime_ns):
                            temporary = preview.with_suffix(".partial.mp4")
                            command = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y",
                                       "-i", str(source), "-an", "-vf", "scale=960:-2",
                                       "-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
                                       "-movflags", "+faststart", str(temporary)]
                            try:
                                subprocess.run(command, check=True, capture_output=True)
                                temporary.replace(preview)
                            except (subprocess.CalledProcessError, OSError) as error:
                                temporary.unlink(missing_ok=True)
                                self.send_error(500, f"Could not make source preview: {error}")
                                return
                    self.send_file(preview, "video/mp4")
                else:
                    file = (trial["replay_video"] if kind == "replay" else
                            trial["guided_replay_video"] if kind == "guided" else
                            trial.get("no_funnel_replay_video", "") if kind == "no_funnel" else
                            trial["fallback_replay_video"] if kind == "fallback" else trial["plot"])
                    self.send_file(Path(file), "image/png" if kind == "plot" else "video/mp4")
            else:
                self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/decision":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 10000:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                trial = by_stem.get(payload.get("stem"))
                if trial is None:
                    raise ValueError("Unknown trial")
                decision = validate_decision(payload, trial)
                with lock:
                    all_decisions = read_decisions(decisions_path)
                    all_decisions[trial["stem"]] = decision
                    write_decisions(decisions_path, all_decisions)
                self.send_json(decision)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, 400)

    return ReviewHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "outputs/dual_marker_original_mount")
    parser.add_argument("--scores", type=Path,
                        default=ROOT / "outputs/evaluation_depth_entry_12mm/results.json")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "outputs/trial_review")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    trials = load_trials(args.results_dir, args.scores)
    decisions = args.review_dir / "decisions.csv"
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 handler_factory(trials, decisions, args.review_dir / "previews"))
    print(f"Reviewing {len(trials)} trials at http://127.0.0.1:{args.port}/", flush=True)
    print(f"Decisions: {decisions}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
