"""Upload the staged datasets and policy to the Hugging Face Hub, or download them.

    python -m mimicvla.dataset.publish.hub upload --owner HF_USER [--raw] [--policy]
    mimicvla download [--raw] [--policy]

Downloads go to data/: the LeRobot training set with its evaluation start
states, optionally the raw recordings and the trained policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

LEROBOT = "mimicvla-panda-usb-15x7"
RAW = "mimicvla-human-usb-raw"
POLICY = "mimicvla-smolvla-usb"
DEFAULT_OWNER = "<HF_USER>"
STAGED = Path("outputs/hf_upload")


def upload(owner: str, raw: bool, policy: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    repos = [(LEROBOT, "dataset", STAGED / LEROBOT)]
    if raw:
        repos.append((RAW, "dataset", STAGED / RAW))
    if policy:
        repos.append((POLICY, "model", STAGED / POLICY))
    for name, kind, folder in repos:
        repo_id = f"{owner}/{name}"
        api.create_repo(repo_id, repo_type=kind, exist_ok=True)
        api.upload_large_folder(repo_id=repo_id, repo_type=kind, folder_path=folder)
        print(f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo_id}")


def download(owner: str = DEFAULT_OWNER, raw: bool = False, policy: bool = False,
             root: Path = Path("data")) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    paths = {"lerobot": snapshot_download(f"{owner}/{LEROBOT}", repo_type="dataset",
                                          local_dir=root / LEROBOT)}
    if raw:
        paths["raw"] = snapshot_download(f"{owner}/{RAW}", repo_type="dataset",
                                         local_dir=root / RAW)
    if policy:
        paths["policy"] = snapshot_download(f"{owner}/{POLICY}", local_dir=root / POLICY)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    up = commands.add_parser("upload")
    up.add_argument("--owner", required=True)
    up.add_argument("--raw", action="store_true", help="also upload the raw recordings (6.8 GB)")
    up.add_argument("--policy", action="store_true", help="also upload the staged policy")
    down = commands.add_parser("download")
    down.add_argument("--owner", default=DEFAULT_OWNER)
    down.add_argument("--raw", action="store_true")
    down.add_argument("--policy", action="store_true")
    args = parser.parse_args()
    if args.command == "upload":
        upload(args.owner, args.raw, args.policy)
    else:
        print(download(args.owner, args.raw, args.policy))


if __name__ == "__main__":
    main()
