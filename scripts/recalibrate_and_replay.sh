#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD/scripts"
export MPLCONFIGDIR="$PWD/outputs/mpl_cache"
mkdir -p "$MPLCONFIGDIR"
PYTHON="$PWD/.venv/bin/python"

"$PYTHON" scripts/recalibrate_recordings.py
"$PYTHON" scripts/replay_recalibrated.py --batch all --geometry both
"$PYTHON" scripts/audit_mount_stability.py
"$PYTHON" scripts/evaluate_marker0_fallbacks.py
"$PYTHON" scripts/audit_guided_contacts.py
"$PYTHON" scripts/build_review_priority.py
"$PYTHON" scripts/summarize_recalibration.py

if [[ "${1:-}" == "--render-review" ]]; then
    MUJOCO_GL=egl "$PYTHON" scripts/render_recalibrated_replays.py --batch dual_original
    MUJOCO_GL=egl "$PYTHON" scripts/render_recalibrated_replays.py --batch dual_original --fallback
    MUJOCO_GL=egl "$PYTHON" scripts/render_recalibrated_replays.py --batch dual_minus3
    MUJOCO_GL=egl "$PYTHON" scripts/render_recalibrated_replays.py --batch single_20mm
fi

echo "Calibration: outputs/recalibration/candidate_mounts.json"
echo "Scores: outputs/recalibration/replay_summary.json"
