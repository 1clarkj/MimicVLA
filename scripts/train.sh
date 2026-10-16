#!/usr/bin/env bash
# Fine-tune SmolVLA on the USB insertion demonstrations. The 8 held-out clips
# (src/mimicvla/dataset/splits.py) are excluded automatically.
#   bash scripts/train.sh RUN_NAME STEPS [DATASET] [BATCH_SIZE]
# DATASET is a dataset folder or a name under outputs/datasets/. Default: the
# downloaded data/mimicvla-panda-usb (`mimicvla download`).
# TRAIN_SUBSET=train16 trains on a nested 16-clip subset (data scaling curve);
# DECAY_STEPS overrides the learning-rate decay horizon (default STEPS).
set -euo pipefail
cd "$(dirname "$0")/.."

usage="Usage: bash scripts/train.sh RUN_NAME STEPS [DATASET] [BATCH_SIZE]"
run_name="${1:?$usage}"
steps="${2:?$usage}"
dataset="${3:-}"
batch_size="${4:-32}"
[[ -n "$dataset" ]] || dataset=data/mimicvla-panda-usb
[[ -d "$dataset" ]] || dataset="outputs/datasets/$dataset"
dataset_dir="$(cd "$dataset" && pwd)"
dataset_id="$(basename "$dataset_dir")"
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
