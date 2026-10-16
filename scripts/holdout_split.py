"""Fixed held-out source clips for 15 x 7 mm policy evaluation.

Drawn once (numpy seed 2026), stratified by recording batch: 4 of 36
single_20mm, 3 of 32 dual_original, 1 of 4 dual_minus3. Keyed by clip so
every dataset built from these recordings holds out the same demonstrations.

    python scripts/holdout_split.py DATASET_DIR [train|holdout|trainN]
prints that split's episode indices as a JSON list. trainN (e.g. train16) is
the first N training episodes of a fixed shuffle (seed 2026), so smaller
subsets are nested in larger ones for the data scaling curve.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from pathlib import Path


HOLDOUT_CLIPS = frozenset({
    ("single_20mm", "20mm_10102026_161008"),
    ("single_20mm", "20mm_10102026_194743"),
    ("single_20mm", "20mm_10102026_195321"),
    ("single_20mm", "20mm_10102026_195559"),
    ("dual_original", "10112026_222734"),
    ("dual_original", "10112026_223000"),
    ("dual_original", "10112026_223447"),
    ("dual_minus3", "10112026_182703"),
})


def split_indices(dataset_dir: Path) -> dict[str, list[int]]:
    manifest = json.loads((dataset_dir / "capture_manifest.json").read_text())
    episodes = manifest["episodes_manifest"]
    holdout = [episode["episode_index"] for episode in episodes
               if (episode["batch"], episode["clip"]) in HOLDOUT_CLIPS]
    train = [episode["episode_index"] for episode in episodes
             if (episode["batch"], episode["clip"]) not in HOLDOUT_CLIPS]
    return {"train": train, "holdout": holdout}


def train_subset(dataset_dir: Path, size: int) -> list[int]:
    train = split_indices(dataset_dir)["train"]
    if not 0 < size <= len(train):
        raise ValueError(f"Subset size must be in 1..{len(train)}")
    order = np.random.default_rng(2026).permutation(len(train))
    return sorted(train[i] for i in order[:size])


if __name__ == "__main__":
    dataset_dir, split = Path(sys.argv[1]), sys.argv[2]
    if split.startswith("train") and split != "train":
        indices = train_subset(dataset_dir, int(split[len("train"):]))
    else:
        indices = split_indices(dataset_dir)[split]
    print(json.dumps(indices).replace(" ", ""))
