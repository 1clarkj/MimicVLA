#!/usr/bin/env bash
# Reproduce the results in RESULTS.md and documentation/experiments.md, in order.
# Each stage can be run on its own:  bash scripts/reproduce_results.sh STAGE
#   data      the training-set variants, rebuilt from the published recordings
#   train     every SmolVLA run (developed on one 16 GB RTX 5060 Ti)
#   evaluate  5 seeds x {72 dataset starts, 55 excluded starts} per policy
#   dsrl      DSRL noise-steering runs on the frozen low-pass policy
#   latency   chunk latency vs denoising steps
#   summary   tables, bootstrap intervals and figures
# Needs `mimicvla download --raw`. The 72-episode training set and the 55
# excluded start states are the published data/mimicvla-panda-usb-15x7.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" MUJOCO_GL=egl
export MENAGERIE_CACHE_DIR="$PWD/outputs/menagerie_cache"
PY=.venv/bin/python
TRAIN=outputs/train
EVAL=outputs/policy_eval_multiseed
RAW=data/mimicvla-human-usb-raw
ALL72=data/mimicvla-panda-usb-15x7
EXC55=$ALL72/eval_starts/excluded55
D=outputs/datasets
LOWPASS=$TRAIN/lowpass_bs32_20k/checkpoints/020000/pretrained_model

build() {  # name [build-dataset options]
    [[ -d $D/$1 ]] || $PY -m mimicvla.cli build-dataset $RAW/tracking --clips $D/clips72.txt \
        --no-videos --output $D/$1 "${@:2}"
}

data() {
    mkdir -p $D
    $PY -c "import json; print(''.join(e['clip'] + '\\n' for e in json.load(open('$ALL72/capture_manifest.json'))['episodes_manifest']), end='')" > $D/clips72.txt
    # The same 72 episodes and actions as the published set (joint states agree
    # to 0.003 rad); the recovery variants need its saved replays.
    build usb_15x7
    build usb_15x7_lowpass2hz --lowpass-hz 2
    build usb_15x7_mirror --mirror
    $PY -m mimicvla.dataset.recovery --copies 4 --correction-gain 0.15 \
        --source-dataset $D/usb_15x7 --dataset-id usb_15x7_recovery
    $PY -m mimicvla.dataset.recovery --copies 4 --correction-gain 0.15 \
        --source-dataset $D/usb_15x7_lowpass2hz --dataset-id usb_15x7_lowpass2hz_recovery
}

train() {
    local t=scripts/train.sh
    bash $t raw_bs32_20k 20000                                   # raw_10k and raw_20k checkpoints
    bash $t lowpass_bs32_20k 20000 usb_15x7_lowpass2hz
    bash $t recovery_v2_bs32_10k 10000 usb_15x7_recovery
    bash $t lowpass_recovery_bs32_20k 20000 usb_15x7_lowpass2hz_recovery
    # data scaling and mirroring: 10k steps on the 20k learning-rate schedule
    TRAIN_SUBSET=train16 DECAY_STEPS=20000 bash $t raw_n16_bs32_10k 10000
    TRAIN_SUBSET=train32 DECAY_STEPS=20000 bash $t raw_n32_bs32_10k 10000
    DECAY_STEPS=20000 bash $t raw_mirror_bs32_10k 10000 usb_15x7_mirror
}

evaluate_policy() {  # name checkpoint [extra args]
    local name=$1 ckpt=$2; shift 2
    for set in all72:$ALL72 excluded55:$EXC55; do for seed in 0 1 2 3 4; do
        local out=$EVAL/$name/${set%%:*}_seed$seed
        [[ -f $out/summary.json ]] && continue
        mkdir -p "$EVAL/$name"
        $PY -m mimicvla.training.evaluate "$ckpt" --dataset-dir "${set#*:}" --split all \
            --n-action-steps 10 --no-video --seed $seed --output-dir "$out" "$@"
    done; done
}

evaluate() {
    local c=checkpoints
    evaluate_policy raw_10k $TRAIN/raw_bs32_20k/$c/010000/pretrained_model
    evaluate_policy raw_20k $TRAIN/raw_bs32_20k/$c/020000/pretrained_model
    evaluate_policy lowpass_20k $LOWPASS
    evaluate_policy recovery_v2_10k $TRAIN/recovery_v2_bs32_10k/$c/010000/pretrained_model
    evaluate_policy lowpass_recovery_20k $TRAIN/lowpass_recovery_bs32_20k/$c/020000/pretrained_model
    evaluate_policy raw_n16_10k $TRAIN/raw_n16_bs32_10k/$c/010000/pretrained_model
    evaluate_policy raw_n32_10k $TRAIN/raw_n32_bs32_10k/$c/010000/pretrained_model
    evaluate_policy raw_mirror_10k $TRAIN/raw_mirror_bs32_10k/$c/010000/pretrained_model
    for n in 1 2 3 5; do
        evaluate_policy lowpass_20k_denoise$n $LOWPASS --num-steps $n
    done
}

dsrl() {
    $PY -m mimicvla.training.dsrl $LOWPASS --output-dir outputs/dsrl/lowpass20k_v1 --decisions 30000
    $PY -m mimicvla.training.dsrl $LOWPASS --output-dir outputs/dsrl/lowpass20k_v2 --decisions 60000 \
        --focus-episodes 8 12 16 18 19 36 38 39 43 44 45 48 51 57 62 66 --focus-prob .5
    for run in zero_noise:zero dsrl_v1:outputs/dsrl/lowpass20k_v1/actor_final.pt \
               dsrl_v2:outputs/dsrl/lowpass20k_v2/actor_final.pt; do
        local name=lowpass_20k_${run%%:*}
        [[ ${run%%:*} == dsrl_* ]] && name=${run%%:*}
        for set in all72:$ALL72 excluded55:$EXC55; do for seed in 0 1 2 3 4; do
            mkdir -p "$EVAL/$name"
            $PY -m mimicvla.training.dsrl $LOWPASS --evaluate "${run#*:}" --dataset-dir "${set#*:}" \
                --split all --seed $seed --output-dir "$EVAL/$name/${set%%:*}_seed$seed"
        done; done
    done
}

latency() {
    $PY -m mimicvla.training.latency $LOWPASS --output outputs/latency/lowpass_20k.json
}

summary() {
    $PY -m mimicvla.training.report
}

stages=("$@")
[[ ${#stages[@]} -eq 0 ]] && stages=(data train evaluate dsrl latency summary)
for stage in "${stages[@]}"; do
    echo "== $stage"
    $stage
done
