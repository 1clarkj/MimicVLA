#!/usr/bin/env bash
# Fine-tune SmolVLA on the 15 x 7 mm no-funnel demonstrations, excluding the
# held-out clips in src/mimicvla/dataset/splits.py.
#   bash scripts/train.sh RUN_NAME STEPS [DATASET_ID] [BATCH_SIZE]
# TRAIN_SUBSET=train16 trains on a nested 16-clip subset (data scaling curve);
# DECAY_STEPS overrides the LR decay horizon (default STEPS).
# DATASET_ID: usb_15x7_no_funnel_full_v2 (raw tracked labels, default) or
#             usb_15x7_no_funnel_lowpass2hz_v1 (low-pass tracked poses)
set -euo pipefail
cd "$(dirname "$0")/.."

usage="Usage: bash scripts/train.sh RUN_NAME STEPS [DATASET_ID] [BATCH_SIZE]"
run_name="${1:?$usage}"
steps="${2:?$usage}"
dataset_id="${3:-usb_15x7_no_funnel_full_v2}"
batch_size="${4:-32}"
dataset_dir="$PWD/outputs/datasets/$dataset_id"
# SmolVLA base weights, pinned to the revision used for every reported run.
export HF_HOME="$PWD/outputs/hf_cache"
model_dir="$(.venv/bin/python -c 'from huggingface_hub import snapshot_download
print(snapshot_download("lerobot/smolvla_base", revision="5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"))')"
output_dir="$PWD/outputs/train/$run_name"

test -f "$dataset_dir/capture_manifest.json"
test -f "$model_dir/model.safetensors"
if [[ -e "$output_dir" ]]; then
    echo "Choose a new run name; output already exists: $output_dir" >&2
    exit 1
fi
train_episodes="$(PYTHONPATH="$PWD/src" .venv/bin/python -m mimicvla.dataset.splits "$dataset_dir" "${TRAIN_SUBSET:-train}")"
echo "Training on episodes $train_episodes"
echo "Held out: $(PYTHONPATH="$PWD/src" .venv/bin/python -m mimicvla.dataset.splits "$dataset_dir" holdout)"

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/lerobot-train \
    --dataset.repo_id="mimicvla/$dataset_id" \
    --dataset.root="$dataset_dir" \
    --dataset.episodes="$train_episodes" \
    --dataset.video_backend=pyav \
    --policy.path="$model_dir" \
    --policy.input_features=null \
    --policy.resize_imgs_with_padding='[256,256]' \
    --policy.n_action_steps=10 \
    --policy.load_vlm_weights=false \
    --policy.scheduler_decay_steps="${DECAY_STEPS:-$steps}" \
    --policy.device=cuda \
    --policy.use_amp=false \
    --policy.push_to_hub=false \
    --output_dir="$output_dir" \
    --batch_size="$batch_size" \
    --num_workers=4 \
    --steps="$steps" \
    --env_eval_freq=0 \
    --save_freq=5000 \
    --wandb.enable=false
