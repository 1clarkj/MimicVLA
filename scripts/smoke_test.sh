#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-outputs/hf_cache}"
export MENAGERIE_CACHE_DIR="${MENAGERIE_CACHE_DIR:-outputs/menagerie_cache}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

.venv/bin/python -m pytest -q
