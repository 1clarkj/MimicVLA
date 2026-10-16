# Scripts

Listed in pipeline order. Generated files go to `outputs/` (not in the
repository). [documentation/pipeline.md](../documentation/pipeline.md)
explains the data stages and
[documentation/experiments.md](../documentation/experiments.md) the policy
experiments. Python scripts are run from the repository root with
`PYTHONPATH=scripts` (some import each other) and, when they use the
simulator, `MUJOCO_GL=egl`.

## Simulation and checks

| Script | Purpose |
|---|---|
| `view_sim.sh` | Open the scene in MuJoCo's interactive viewer. |
| `smoke_test.sh` | Simulation, vision and dataset tests in one go. |
| `replay_demo.sh` | Replay one tracked recording on the Panda. |

## 1. Tracking the recordings

| Script | Purpose |
|---|---|
| `evaluate_recordings.sh` | Track and replay the single-marker recordings. |
| `process_new_recordings.sh` | Track a batch of two-marker recordings and score their replays. |

## 2. Recalibration, replay and review

| Script | Purpose |
|---|---|
| `recalibrate_and_replay.sh` | Runs the stage below end to end. |
| `recalibrate_recordings.py` | Fit per-session marker-to-tip offsets from frames where the plug is inside the port. |
| `replay_recalibrated.py` | Replay every track with the fitted offsets on the strict and widened ports. |
| `audit_mount_stability.py` | Flag clips where the two USB markers disagree (mount moved or tracking ambiguous). |
| `evaluate_marker0_fallbacks.py` | Retry with marker 0 alone where the fused track misses. |
| `audit_guided_contacts.py` | Contact force up to first insertion, shown in the review dashboard. |
| `build_review_priority.py`, `summarize_recalibration.py` | Review queue and replay summary. |
| `render_recalibrated_replays.py` | Robot replay videos for review. |
| `review_trials.sh` / `review_trials.py` / `trial_review.html` | Local review dashboard: original video beside its robot replay; stores accept/reject decisions. |

## 3. Selection and the training dataset

| Script | Purpose |
|---|---|
| `curate_wide_no_funnel.py`, `replay_additional_wide_no_funnel.py`, `curate_additional_wide_no_funnel.py` | Review queues for the widened, funnel-free port (two-marker, then single-marker and older clips). |
| `export_recalibrated_lerobot.py` | Shared LeRobot exporter; produced the 103 accepted clips on the 18 × 10 mm review port. |
| `sweep_no_funnel_port.py` | Replay the 103 accepted clips against 46 port sizes. |
| `audit_compact_full_replays.py` | Check each selected clip's whole replay, not only up to 7 mm depth. |
| `export_compact_no_funnel_lerobot.py` / `prepare_15x7_dataset.sh` | The 72-episode 15 × 7 mm training dataset (`usb_15x7_no_funnel_full_v2`). |
| `holdout_split.py` | The fixed 8-clip holdout and the nested 16/32-clip training subsets. |
| `build_excluded_starts.py` | The 55 excluded-start evaluation set. |

## 4. Label variants

| Script | Purpose |
|---|---|
| `export_smoothed_lerobot.py` | 2 Hz low-pass tracked poses, re-retargeted and re-replayed. |
| `export_recovery_lerobot.py` | DART-style recovery copies (`--correction-gain 0.15` for the gentle labels). |
| `export_mirrored_lerobot.py` | Human trajectories mirrored across the port, re-retargeted and re-rendered. |

## 5. Training and evaluation

| Script | Purpose |
|---|---|
| `train_no_funnel_smolvla.sh` | Fine-tune SmolVLA (holdout clips excluded automatically). |
| `evaluate_policy.py` | Closed-loop rollouts from a set of start states. |
| `dsrl_smolvla.py` | DSRL: SAC over the initial flow-matching noise of a frozen SmolVLA. |
| `benchmark_latency.py` | Chunk latency per denoising-step setting. |
| `summarize_multiseed.py`, `summarize_overnight.py` | Tables, bootstrap intervals and figures. |
| `reproduce_results.sh` | Every export, training run, evaluation and summary behind the reported numbers. |

## 6. Publishing

| Script | Purpose |
|---|---|
| `build_readme_videos.py` | Side-by-side videos: iPhone tracking, robot replay, policies. |
| `build_documentation_examples.py` | Short evidence clips of tracking problems. |
| `export_hf_raw.py` | Hugging Face dataset of all human recordings with tracking and labels. |
