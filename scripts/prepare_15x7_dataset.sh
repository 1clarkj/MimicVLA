#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL=egl
export HF_HOME="$PWD/outputs/hf_cache"
export MENAGERIE_CACHE_DIR="$PWD/outputs/menagerie_cache"
export PYTHONPATH="$PWD/src"

.venv/bin/python scripts/export_compact_no_funnel_lerobot.py
.venv/bin/python -m mimicvla.cli validate-sim-dataset \
    --root outputs/datasets/usb_15x7_no_funnel_full_v2
