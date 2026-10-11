#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MENAGERIE_CACHE_DIR=outputs/menagerie_cache
export MUJOCO_GL=egl
export UV_CACHE_DIR=outputs/uv_cache

exec uv run --locked mimicvla replay-tracking \
  --trajectory outputs/demo_20mm_161008/trajectory.jsonl \
  --output-dir outputs/replay_20mm_161008_full \
  --complete-segment
