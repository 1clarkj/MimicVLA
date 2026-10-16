"""The review queue preserves decisions and marks deliberate quality overrides."""

import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "review_trials", Path(__file__).resolve().parents[1] / "scripts/review_trials.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def test_trial_queue_prioritizes_replay_quality_regardless_of_simulated_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "ROOT", tmp_path)
    batch = tmp_path / "outputs/dual_marker_original_mount"
    batch.mkdir(parents=True)
    video_root = "video_data/double_marker_original_mount/"
    def result(stem, mode, eligible, quality=True):
        return {"video": video_root + stem + ".mov", "mode": mode, "frames": 100,
                "valid_fraction": .8, "longest_raw_tracking_gap_frames": 4,
                "interpolated_frames": 2, "implausible_pose_steps": 0,
                "marker_agreement_mm_p50": 2, "marker_agreement_mm_p95": 8,
                "marker_disagreement_over_5mm_frames": 3, "eligible": eligible,
                "replay_quality_ok": quality}
    (batch / "results.json").write_text(json.dumps([
        result("10112026_222530", "dual", True),
        result("10112026_222522", "dual", True),
        result("10112026_222522", "marker0_only", False),
    ]))
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps([
        {"batch": "dual_marker_original_mount", "mode": "dual",
         "video": video_root + "10112026_222522.mov", "episode_success": True,
         "replay_quality_ok": True},
        {"batch": "dual_marker_original_mount", "mode": "dual",
         "video": video_root + "10112026_222530.mov", "episode_success": False,
         "replay_quality_ok": True},
    ]))
    trials = review.load_trials(batch, scores)
    assert [trial["stem"] for trial in trials] == ["10112026_222522", "10112026_222530"]
    assert trials[0]["marker0_eligible"] is False
    assert trials[0]["priority"] == 0
    assert trials[1]["priority"] == 0  # Simulated miss does not lower review priority.


def test_accept_override_records_failed_checks_and_reason(tmp_path):
    trial = {"stem": "clip", "eligible": True, "success": True, "quality_ok": True}
    form = {"stem": "clip", "decision": "accept", "human_success": "yes",
            "tracking": "good", "replay": "good", "notes": "Observed insertion"}
    decision = review.validate_decision(form, trial)
    path = tmp_path / "decisions.csv"
    review.write_decisions(path, {"clip": decision})
    assert review.read_decisions(path)["clip"]["notes"] == "Observed insertion"
    assert review.read_decisions(path)["clip"]["override"] == "False"
    simulated_miss = review.validate_decision(form, {**trial, "success": False})
    assert simulated_miss["decision"] == "accept"
    assert simulated_miss["override"] == "False"
    with pytest.raises(ValueError, match="Manual override"):
        review.validate_decision({**form, "tracking": "questionable"}, trial)
    assert review.validate_decision(form, {**trial, "quality_ok": False})["override"] == "False"
    with pytest.raises(ValueError, match="Add a reason"):
        review.validate_decision({**form, "tracking": "bad", "override": True,
                                  "notes": ""}, trial)
    overridden = review.validate_decision({**form, "tracking": "bad", "override": True,
                                            "notes": "Source insertion visible; replay remains usable"},
                                           {**trial, "quality_ok": False})
    assert overridden["override"] == "True"
    assert overridden["override_reasons"] == "tracking_bad"
    review.write_decisions(path, {"clip": overridden})
    assert review.read_decisions(path)["clip"]["override_reasons"] == overridden["override_reasons"]
    visually_accepted = review.validate_decision(
        {**form, "tracking": "questionable", "replay": "questionable",
         "override": True, "notes": "Visible successful insertion; pose labels need separate review"},
        {**trial, "success": False, "quality_ok": False})
    assert visually_accepted["decision"] == "accept"
    assert "automated_insertion" not in visually_accepted["override_reasons"]
    with pytest.raises(ValueError, match="no_replayable_span"):
        review.validate_decision(form, {**trial, "eligible": False})
    assert review.validate_decision({**form, "decision": "hold", "tracking": "questionable"},
                                    trial)["decision"] == "hold"


def test_existing_decisions_survive_csv_schema_update(tmp_path):
    path = tmp_path / "decisions.csv"
    path.write_text("stem,decision,human_success,tracking,replay,notes,updated_at\n"
                    "old,accept,yes,good,good,,2026-10-12T15:00:00+00:00\n")
    decisions = review.read_decisions(path)
    assert decisions["old"]["override"] == "False"
    review.write_decisions(path, decisions)
    assert review.read_decisions(path)["old"]["decision"] == "accept"
