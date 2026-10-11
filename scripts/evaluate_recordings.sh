#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MENAGERIE_CACHE_DIR=outputs/menagerie_cache
export MUJOCO_GL=egl
export UV_CACHE_DIR=outputs/uv_cache
export MPLCONFIGDIR=outputs/matplotlib_cache

exec uv run --locked mimicvla evaluate-recordings \
  --video-dir video_data \
  --calibration outputs/iphone_calibration.json \
  --output-dir outputs/video_evaluation
