"""The data-scaling subsets are nested and exclude the holdout clips."""

import json

from mimicvla.dataset.splits import HOLDOUT_CLIPS, split_indices, train_subset


def test_train_subsets_are_nested_and_exclude_holdout(tmp_path):
    clips = ([f"clip_{i:02d}" for i in range(20)] + sorted(HOLDOUT_CLIPS) +
             [f"{clip}_mirror" for clip in sorted(HOLDOUT_CLIPS)])
    manifest = {"episodes_manifest": [{"episode_index": i, "clip": c}
                                      for i, c in enumerate(clips)]}
    (tmp_path / "capture_manifest.json").write_text(json.dumps(manifest))
    train = set(split_indices(tmp_path)["train"])
    assert len(train) == 20
    small, large = train_subset(tmp_path, 5), train_subset(tmp_path, 10)
    assert set(small) < set(large) <= train
    assert train_subset(tmp_path, 20) == sorted(train)
