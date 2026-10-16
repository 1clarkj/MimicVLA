#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VIDEO_DIR=video_data/double_marker_original_mount
PYTHON_BIN=.venv/bin/python

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing .venv. Run: uv sync --python 3.12 --group train" >&2
    exit 1
fi

shopt -s nullglob
videos=("$VIDEO_DIR"/*.mov "$VIDEO_DIR"/*.mp4)
if (( ${#videos[@]} == 0 )); then
    echo "No .mov or .mp4 recordings in $VIDEO_DIR" >&2
    exit 1
fi

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "Would process ${#videos[@]} videos from $VIDEO_DIR"
    echo "Output: outputs/dual_marker_original_mount/"
    echo "Current scores: outputs/evaluation_depth_entry_12mm/"
    echo "The evaluator reruns all videos and refits the marker mount."
    exit 0
fi
if (( $# > 0 )); then
    echo "Usage: bash scripts/process_new_recordings.sh [--dry-run]" >&2
    exit 2
fi

export MENAGERIE_CACHE_DIR="$PWD/outputs/menagerie_cache"
export MUJOCO_GL=egl
export MPLCONFIGDIR="$PWD/outputs/mpl_cache"
mkdir -p "$MENAGERIE_CACHE_DIR" "$MPLCONFIGDIR"

"$PYTHON_BIN" -m mimicvla.cli evaluate-dual-recordings \
    --video-dir "$VIDEO_DIR" \
    --calibration outputs/iphone_calibration.json
"$PYTHON_BIN" -m mimicvla.cli rescore-replays \
    --output-dir outputs/evaluation_depth_entry_12mm

echo "Inspect outputs/evaluation_depth_entry_12mm/summary.json and results.csv"
echo "Inspect outputs/dual_marker_original_mount/dual/<video-stem>/replay/replay.mp4"
