"""Stage the trained SmolVLA policy for the Hugging Face Hub with a model card.

    python -m mimicvla.dataset.publish.policy [--owner HF_USER] [--checkpoint DIR]
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

CHECKPOINT = Path("outputs/train/raw_bs32_20k/checkpoints/010000/pretrained_model")
REPO = "mimicvla-smolvla-usb"

CARD = """---
license: mit
library_name: lerobot
pipeline_tag: robotics
base_model: lerobot/smolvla_base
datasets:
- {owner}/mimicvla-panda-usb-15x7
tags:
- smolvla
- franka-panda
- mujoco
- usb-insertion
---

# SmolVLA fine-tuned for USB insertion in MuJoCo

`lerobot/smolvla_base` fine-tuned for 10k steps (batch 32) on the 64 training
episodes of [`{owner}/mimicvla-panda-usb-15x7`](https://huggingface.co/datasets/{owner}/mimicvla-panda-usb-15x7),
which are human phone demonstrations replayed on a simulated Franka Panda.

Inputs: fixed front and wrist RGB (256 × 256), 9-D Panda state and the
instruction "Insert the USB stick into the USB port." Output: chunks of 50
six-dimensional end-effector actions at 20 Hz, of which 10 are executed
before re-planning.

Closed-loop success in the simulator (5 seeds per start, 95% bootstrap
intervals over starts):

| Start set | Success |
|---|---|
| 64 training starts | 60% [52, 68] |
| 55 starts from recordings never trained on | 49% [39, 59] |

The port is a 15 × 7 mm opening (the real one is 13.4 × 5.9 mm). Simulation
only; not tested on hardware. Code and evaluation:
[github.com/1clarkj/MimicVLA](https://github.com/1clarkj/MimicVLA).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="<HF_USER>")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=Path("outputs/hf_upload") / REPO)
    args = parser.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    shutil.copytree(args.checkpoint, args.output)
    # Replace local paths in the saved configs with portable ones.
    root = str(Path.cwd()) + "/"
    snapshot = re.compile(r'"[^"]*models--lerobot--smolvla_base/snapshots/[0-9a-f]+"')
    for name in ("config.json", "train_config.json"):
        path = args.output / name
        text = snapshot.sub('"lerobot/smolvla_base"', path.read_text()).replace(root, "")
        path.write_text(text)
    (args.output / "README.md").write_text(CARD.replace("{owner}", args.owner))
    print(f"Staged {args.output}")


if __name__ == "__main__":
    main()
