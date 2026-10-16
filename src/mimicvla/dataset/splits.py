"""Fixed held-out clips of the published recordings for policy evaluation.

Drawn once (numpy seed 2026), stratified by recording session: 4 of 36
single-marker clips, 3 of 32 and 1 of 4 two-marker clips. Keyed by clip so
every dataset built from these recordings holds out the same demonstrations.

    python -m mimicvla.dataset.splits DATASET_DIR [train|holdout|trainN]
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
    "20mm_10102026_161008", "20mm_10102026_194743", "20mm_10102026_195321",
    "20mm_10102026_195559", "10112026_222734", "10112026_223000", "10112026_223447",
    "10112026_182703",
})


def _held_out(clip: str) -> bool:
    """Mirrored copies (``<clip>_mirror``) are held out with their source clip."""
    return clip.removesuffix("_mirror") in HOLDOUT_CLIPS


def split_indices(dataset_dir: Path) -> dict[str, list[int]]:
    manifest = json.loads((dataset_dir / "capture_manifest.json").read_text())
    episodes = manifest["episodes_manifest"]
    holdout = [episode["episode_index"] for episode in episodes if _held_out(episode["clip"])]
    train = [episode["episode_index"] for episode in episodes if not _held_out(episode["clip"])]
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
