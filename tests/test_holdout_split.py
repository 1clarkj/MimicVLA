"""The data-scaling subsets are nested and exclude the holdout clips."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from holdout_split import HOLDOUT_CLIPS, split_indices, train_subset  # noqa: E402


def test_train_subsets_are_nested_and_exclude_holdout(tmp_path):
    clips = [("single_20mm", f"clip_{i:02d}") for i in range(20)] + sorted(HOLDOUT_CLIPS)
    manifest = {"episodes_manifest": [{"episode_index": i, "batch": b, "clip": c}
                                      for i, (b, c) in enumerate(clips)]}
    (tmp_path / "capture_manifest.json").write_text(json.dumps(manifest))
    train = set(split_indices(tmp_path)["train"])
    assert len(train) == 20
    small, large = train_subset(tmp_path, 5), train_subset(tmp_path, 10)
    assert set(small) < set(large) <= train
    assert train_subset(tmp_path, 20) == sorted(train)
