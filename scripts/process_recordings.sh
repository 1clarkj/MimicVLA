#!/usr/bin/env bash
# Turn the iPhone recordings in video_data/ into the 72-episode training set.
#   bash scripts/process_recordings.sh track    marker tracking, recalibration, robot replays, review queues
#   (review)  python -m mimicvla.dataset.review  -> accept/reject each clip in the browser
#   bash scripts/process_recordings.sh select   port sweep, full-replay audit, LeRobot export, eval sets
# documentation/pipeline.md describes each step.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" MUJOCO_GL=egl
export MENAGERIE_CACHE_DIR="$PWD/outputs/menagerie_cache" MPLCONFIGDIR="$PWD/outputs/mpl_cache"
PY=.venv/bin/python
CAL=outputs/iphone_calibration.json

track() {
    # Marker poses per frame (the plug-tip offset is applied later, per session)
    $PY -m mimicvla.cli evaluate-recordings --video-dir video_data --calibration $CAL \
        --output-dir outputs/video_evaluation_pose_prior
    $PY -m mimicvla.cli evaluate-dual-recordings --video-dir video_data/double_marker_original_mount \
        --calibration $CAL
    $PY -m mimicvla.cli evaluate-dual-recordings --video-dir video_data/double_marker \
        --calibration $CAL --tip-x-mm 0 --output-dir outputs/dual_marker_comparison
    $PY -m mimicvla.cli evaluate-dual-recordings --video-dir video_data/double_marker \
        --calibration $CAL --reuse-tracking-from outputs/dual_marker_comparison
    $PY -m mimicvla.dataset.rescore    # 12 mm scores shown in the review dashboard
    # Per-session marker-to-tip offsets, then robot replays on the real and widened ports
    $PY -m mimicvla.dataset.recalibrate
    $PY -m mimicvla.dataset.replay --batch all
    $PY -m mimicvla.dataset.render --batch dual_original
    $PY -m mimicvla.dataset.replay_additional --batch all
    $PY -m mimicvla.dataset.curate
}

select_clips() {
    $PY -m mimicvla.dataset.export_review
    $PY -m mimicvla.dataset.port_sweep
    $PY -m mimicvla.dataset.audit   # reads the 75-clip intermediate export; see documentation/pipeline.md
    $PY -m mimicvla.dataset.export
    $PY -m mimicvla.cli validate-sim-dataset --root outputs/datasets/usb_15x7_no_funnel_full_v2
    $PY -m mimicvla.dataset.excluded_starts
}

case "${1:-}" in
    track) track ;;
    select) select_clips ;;
    *) echo "Usage: bash scripts/process_recordings.sh track|select" >&2; exit 2 ;;
esac
