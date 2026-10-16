#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MENAGERIE_CACHE_DIR=outputs/menagerie_cache
export MUJOCO_GL=egl
export MPLCONFIGDIR=outputs/matplotlib_cache

exec .venv/bin/python -m mimicvla.cli evaluate-recordings \
  --video-dir video_data \
  --calibration outputs/iphone_calibration.json \
  --output-dir outputs/video_evaluation_12mm_full_replay
