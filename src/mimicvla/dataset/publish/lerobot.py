"""Stage the LeRobot training set for the Hugging Face Hub.

Copies the 72-episode dataset and adds what evaluation needs without the
local replays: each episode's alignment actions (reset to start pose) in
capture_manifest.json, and the 55 excluded start states in
eval_starts/excluded55/. Writes the dataset card. Nothing is uploaded.

    python -m mimicvla.dataset.publish.lerobot [--owner HF_USER]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mimicvla.training.evaluate import alignment_actions

DATASET = Path("outputs/datasets/usb_15x7_no_funnel_full_v2")
EXCLUDED = Path("outputs/eval_sets/excluded_starts_v1")
REPO = "mimicvla-panda-usb-15x7"


def with_actions(manifest: dict) -> dict:
    episodes = []
    for episode in manifest["episodes_manifest"]:
        episode = {key: value for key, value in episode.items() if key != "replay_dir"}
        episodes.append({**episode, "alignment_actions": alignment_actions(episode)})
    return {**manifest, "episodes_manifest": episodes}


def stage(output: Path, owner: str) -> None:
    if output.exists():
        shutil.rmtree(output)
    for name in ("data", "meta", "videos"):
        shutil.copytree(DATASET / name, output / name)
    manifest = json.loads((DATASET / "capture_manifest.json").read_text())
    (output / "capture_manifest.json").write_text(json.dumps(with_actions(manifest), indent=2) + "\n")
    excluded = json.loads((EXCLUDED / "capture_manifest.json").read_text())
    (output / "eval_starts/excluded55").mkdir(parents=True)
    (output / "eval_starts/excluded55/capture_manifest.json").write_text(
        json.dumps(with_actions(excluded), indent=2) + "\n")
    (output / "README.md").write_text(CARD.replace("{owner}", owner))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="<HF_USER>", help="Hugging Face user or organisation")
    parser.add_argument("--output", type=Path, default=Path("outputs/hf_upload") / REPO)
    args = parser.parse_args()
    stage(args.output, args.owner)
    print(f"Staged {args.output}")


CARD = """---
license: mit
task_categories:
- robotics
tags:
- LeRobot
- franka-panda
- mujoco
- usb-insertion
- human-demonstration
- smolvla
pretty_name: MimicVLA Panda USB insertion (15 x 7 mm, MuJoCo)
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# MimicVLA: Panda USB insertion from human phone recordings

72 episodes of a simulated Franka Panda inserting a USB plug into a wall port
in MuJoCo. Every episode is a **human demonstration**: a person's plug motion,
tracked with ArUco markers in a handheld iPhone video, was retargeted to the
Panda and replayed. The images, joint states and actions were recorded during
that replay.

- Code: [github.com/1clarkj/MimicVLA](https://github.com/1clarkj/MimicVLA)
- Source videos and tracking: [`{owner}/mimicvla-human-usb-raw`](https://huggingface.co/datasets/{owner}/mimicvla-human-usb-raw).
  Its `lerobot_episode_index` column maps each recording to an episode here.

## Format

LeRobot v3.0, 20 Hz, 8,089 frames, one task:
`"Insert the USB stick into the USB port."`

| Feature | Shape | Meaning |
|---|---|---|
| `observation.images.camera1` | 256 × 256 × 3 | Fixed front camera looking at the port |
| `observation.images.camera2` | 256 × 256 × 3 | Wrist camera on the Panda hand |
| `observation.state` | 9 | 7 arm joint angles (rad) and 2 finger positions (m), measured before the action |
| `action` | 6 | `[dx, dy, dz, rx, ry, rz]`: world-frame end-effector translation (m) and rotation vector (rad) per step |

The action is applied to the end-effector target; differential IK (Mink) and
the Panda's position actuators track it. The plug is rigidly attached to the
hand.

## Task and selection

The port is a 15 × 7 mm rectangular opening without a funnel in a flush wall;
the plug is 12 × 4.5 mm. The real port is 13.4 × 5.9 mm. The opening was
widened because marker-tracking error (1–3 mm) exceeds the real clearance
(0.7 mm per side). Success means the plug-tip centre is inside the opening
and at least 12 mm deep.

Episodes start after a short simulator alignment to the first tracked pose
and run to the end of the human motion. Each was accepted in manual review,
and its replay reaches 7 mm depth during the human motion, finishes at least
12 mm deep, and has no deep collision or joint-limit fault.
`capture_manifest.json` records each episode's source clip, frame range and
peak contact force.

## Splits

The policies in the project were trained on 64 episodes and held out these 8
(stratified by recording batch, seed 2026):

```
[0, 6, 23, 28, 47, 55, 67, 70]
```

## Evaluation start states

`capture_manifest.json` lists each episode's source clip. Its
`alignment_actions` move the simulated arm from reset to the episode's start
pose, so a policy can be evaluated from exactly the demonstrated starts.
`eval_starts/excluded55/capture_manifest.json` holds 55 further start poses
from recordings that are not in this dataset; they are the main
generalisation test in the project.

## Usage

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("{owner}/mimicvla-panda-usb-15x7")
frame = dataset[0]
```

With the code repository, `uv run mimicvla download` fetches this dataset
into `data/`.

## Limitations

Simulation only. Images are rendered, with fixed lighting. One person, one
port pose. Contact forces in the replays are often high (peak per episode is
in `capture_manifest.json`).
"""


if __name__ == "__main__":
    main()
