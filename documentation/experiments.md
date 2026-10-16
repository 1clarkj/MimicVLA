# Experiments

All numbers are closed-loop MuJoCo rollouts of fine-tuned SmolVLA policies,
not demonstration replays. The story and the main conclusions are in
[RESULTS.md](../RESULTS.md); this page has the full tables.

## Protocol

`mimicvla.training.evaluate` starts each rollout from a recorded start state:
it replays that clip's simulator alignment prefix, then hands control to the
policy. A rollout succeeds when the USB tip centre is inside the
15 × 7 mm opening and at least 12 mm deep within 400 steps (20 s).
"Deep collision" means the simulator's `bad_collision` flag (more than 1 mm
penetration into the port walls) fired at least once. By default the policy
executes 10 of each 50-action chunk before re-planning (`--n-action-steps`).

Start sets:

- **Training starts (64)**: the clips of the 72-episode training set
  (`1clarkj/mimicvla-panda-usb-15x7`) used for training.
- **Holdout (8)**: clips of the same dataset that no model trained on
  (`mimicvla.dataset.splits`, seed 2026, stratified by recording session).
  Too few to rank models.
- **Excluded starts (55)**: aligned start poses from recordings that never
  entered any training set (16 failed the 15 × 7 mm replay check, 7 were
  rejected or held in review, 16 were unreviewed or never reviewed, and 16 others). Only the start
  pose is used. **This is the main test of generalisation.**

Sections 1–5 use 5 seeds per start. A start's success is its mean over seeds;
intervals are 95% bootstraps over start states, and differences are paired
over the same starts. Section 7 records the earlier single-rollout
experiments that led to this protocol.

## Training runs

| Run | Labels | Train clips | Batch | Steps |
|---|---|---|---|---|
| `raw_10k`, `raw_20k` | raw tracked poses | 64 | 32 | 10k / 20k checkpoints of one run |
| `lowpass_20k` | 2 Hz low-pass tracked poses | 49 (17 clips no longer insert after filtering) | 32 | 20k |
| `recovery_v2_10k` | raw + gentle recovery copies (gain 0.15) | 64 | 32 | 10k |
| `lowpass_recovery_20k` | low-pass + gentle recovery copies | 49 | 32 | 20k |
| `raw_n16_10k`, `raw_n32_10k` | raw, nested subsets | 16 / 32 | 32 | 10k (20k LR schedule) |
| `raw_mirror_10k` | raw + mirrored trajectories | 110 | 32 | 10k (20k LR schedule) |
| `dsrl_v1`, `dsrl_v2` | frozen `lowpass_20k` + SAC noise actor | 64 | – | 30k / 60k decisions |

All start from `lerobot/smolvla_base` and use both cameras, the 9-D Panda
state and the 6-D Cartesian action.


## 1. Final comparison

Every policy ran 5 seeds from each start. Success is the mean over starts of
each start's success rate, with a 95% bootstrap interval over start states.
"Excluded starts" are 55 aligned start states from recordings that never
entered any training set (`eval_starts/excluded55` in the published dataset): 16 failed
the 15 x 7 mm replay check, 7 were rejected or held in review, 3 unreviewed,
13 never reviewed, and 16 others. Only their start pose is used, so their
tracking quality after the start does not matter.

| Policy | Training starts (64) | Holdout (8) | Excluded starts (55) | Deep-collision rollouts (64 / 55) | Median peak force (64 / 55) |
|---|---|---|---|---|---|
| `raw_10k` | 60% [52, 68] | 57% [30, 82] | 49% [39, 59] | 3% / 4% | 172 / 218 N |
| `raw_20k` | 55% [49, 62] | 55% [32, 78] | 43% [33, 52] | 6% / 4% | 197 / 281 N |
| `lowpass_20k` | 82% [74, 89] | 62% [25, 88] | 34% [23, 47] | 18% / 61% | 128 / 415 N |
| `lowpass_20k_zero_noise` | 81% [72, 89] | 60% [25, 88] | 33% [21, 45] | 19% / 64% | 128 / 417 N |
| `dsrl_v1` | 80% [70, 88] | 62% [25, 88] | 32% [21, 44] | 19% / 64% | 131 / 431 N |
| `dsrl_v2` | 82% [74, 90] | 62% [25, 88] | 35% [23, 47] | 18% / 61% | 121 / 439 N |
| `lowpass_recovery_20k` | 66% [58, 75] | 43% [18, 68] | 52% [42, 62] | 34% / 44% | 161 / 216 N |

Paired differences (same starts, 95% bootstrap interval):

- Low-pass vs raw 10k, dataset starts: **+20.3 pts** [+11.7, +28.9].
- Raw 10k vs low-pass, excluded starts: **+14.9 pts** [+4.0, +26.2].
- Low-pass + gentle recovery vs low-pass, excluded starts: **+18.2 pts** [+6.5, +29.5].
- DSRL v2 vs low-pass (random noise), dataset starts: +0.6 pts [-4.4, +5.3].

Interpretation:

1. **Low-pass labels overfit the demonstrated starts.** They give the best
   numbers on the 72 dataset starts but the worst on unseen starts, with deep
   collisions in 61% of those rollouts and median peak force above 400 N. The
   raw-label model is less accurate on the demonstrated starts but degrades
   less on new ones.
2. **Recovery data does what DART is meant to do, just not where we first
   measured it.** On the dataset starts it costs about 16 pts; on unseen starts
   it is the best policy (52%) and roughly halves the low-pass model's force.
   This is the classic robustness trade-off of noise-injected demonstrations.
3. **DSRL did not help.** Neither run (v2 sampled the 16 training starts whose
   outcome flipped between runs half the time, 60k decisions) moved success,
   collisions or force beyond seed noise. The base policy's errors on unseen
   starts are systematic, not a matter of an unlucky noise sample, and noise
   steering cannot reach them.
4. **The 8-clip holdout cannot rank these policies.** Its intervals span
   about 60 pts; the 55 excluded starts are the informative test.



## 2. Data scaling

Same recipe as `raw_10k` (raw labels, batch 32, 10k steps, learning-rate
schedule of the 20k run so the 64-demo point is the same checkpoint).
`mimicvla.dataset.splits` draws nested subsets of the 64 training clips
(seed 2026): 16 ⊂ 32 ⊂ 64.

| Demos | Policy | Trained starts | Unseen dataset starts | Excluded starts (55) | Deep-collision rollouts (55) | Median peak force (55) |
|---|---|---|---|---|---|---|
| 16 | `raw_n16_10k` | 48% [31, 64] | 19% [11, 28] (n=56) | 16% [9, 24] | 75% | 610 N |
| 32 | `raw_n32_10k` | 45% [34, 56] | 21% [12, 31] (n=40) | 24% [16, 32] | 21% | 285 N |
| 64 | `raw_10k` | 60% [52, 68] | 57% [30, 82] (n=8) | 49% [39, 59] | 4% | 218 N |
| 110 (64 + 46 mirrored) | `raw_mirror_10k` | 47% [40, 54] | 48% [25, 70] (n=8) | 40% [32, 49] | 17% | 222 N |

Paired differences on the 55 excluded starts:

- 64 vs 32 demos: **+25.1 pts** [+14.9, +35.6].
- 64 vs 16 demos: **+33.1 pts** [+21.8, +44.0].
- 32 vs 16 demos: +8.0 pts [−0.4, +16.7].

Interpretation:

- **More human demonstrations help, and the curve has not flattened.** The
  gain from 32 to 64 is larger than from 16 to 32. Success on unseen starts
  roughly doubles and deep collisions drop from 75% to 4% of rollouts. The
  64-demo model generalises almost as well as it fits.
- **At 16 and 32 demos the policy memorises.** It reproduces about half of
  its own starts but only about a fifth of new ones. With 64 demos, the
  trained-start and unseen-start rates nearly meet.
- **The 8-demo point was not run.** Its training job ran out of GPU memory
  while three evaluations shared the GPU, and there was no time to restart it.
- The obvious next step is more recordings. Each one is a human trajectory,
  so this is the cheapest lever available.


## 3. Mirror augmentation: valid, but it did not help

**Why not flip images.** The scene is not left–right symmetric. The fixed
camera sits to one side of the port, and the wrist camera is mounted 45 mm to
one side of the plug. A flipped image is therefore a view that no camera in
the scene can produce. The 9-D joint-state observation cannot be mirrored
either, because a Panda configuration has no mirrored counterpart through a
simple sign flip. Image flipping would teach the policy inconsistent data.

**What was done instead.** `retarget.py` gained a `mirror_socket_x` option.
It reflects the tracked human trajectory across the port's vertical
mid-plane (p → Mp, R → MRM with M = diag(−1, 1, 1) in the socket frame,
which keeps rotations proper). The mirrored human motion is then
re-retargeted onto the Panda and re-rendered from the real cameras.
Every mirrored episode is a physically valid robot demonstration with correct
images, joint states and actions. It still comes from a human trajectory.

- `mimicvla build-dataset --mirror` → `usb_15x7_mirror`.
  Of 72 mirrored clips, **48 pass the same 15 × 7 mm replay gate** as the
  originals. Almost all 24 rejects stall on the rim at about 0 mm depth, the
  same failure as low-pass breakage. A plausible reason is that the
  human's fine corrections are not symmetric about the opening. The result is 120 episodes and 13,806 frames.
- Mirrors of holdout clips stay in the holdout, so 110 episodes train
  (64 originals + 46 mirrors).
- **Applied to raw labels**, the data-scaling family. Raw labels give the
  best policy on unseen starts without recovery data. `lowpass_recovery_20k`
  (52%) and `raw_10k` (49%) are within noise of each other. Raw labels also keep
  all 72 clips, while low-pass labels break 17.

**Result.** `raw_mirror_10k` is worse than `raw_10k`: −8.7 pts on excluded
starts [−18.2, +0.7] and −12.5 pts on training starts [−22.2, −3.1]. Deep
collisions on excluded starts rise from 4% to 17%.

Likely reasons, untested:

1. **Fixed step budget.** Both runs train for 10k steps, so with 110 episodes
   each original clip is seen about 40% less often. The training-start drop
   is consistent with underfitting.
2. **Mirrored starts may not match test starts.** If real start states
   cluster on one side of the port (one person, one approach habit), mirrored
   data spends capacity on the other side, which the excluded starts rarely
   visit. The start distribution has not been checked for this.

A 20k-step mirror run would separate these, but was not run. For now, real
recordings beat synthetic mirrors.

**Lighting augmentation was skipped.** The simulator renders with fixed
lighting, so evaluation in sim cannot show a benefit. It belongs to
sim-to-real transfer, which this project does not test.


## 4. Labels × recovery data (2 × 2)

![Labels × recovery](figures/labels_x_recovery.png)

The missing cell, raw labels + gentle recovery (`recovery_v2_10k`), now has
5 seeds.

| Labels | Recovery data | Policy | Training starts (64) | Holdout (8) | Excluded starts (55) | Deep-collision rollouts (64 / 55) | Median peak force (64 / 55) |
|---|---|---|---|---|---|---|---|
| raw | none | `raw_10k` | 60% [52, 67] | 57% [30, 82] | 49% [39, 59] | 3% / 4% | 172 / 218 N |
| raw | gentle recovery | `recovery_v2_10k` | 55% [48, 61] | 48% [30, 68] | 51% [44, 56] | 16% / 14% | 189 / 213 N |
| low-pass | none | `lowpass_20k` | 82% [74, 89] | 62% [25, 88] | 34% [23, 46] | 18% / 61% | 128 / 415 N |
| low-pass | gentle recovery | `lowpass_recovery_20k` | 66% [57, 75] | 43% [18, 68] | 52% [43, 62] | 34% / 44% | 161 / 216 N |

Paired differences on the excluded starts: raw + recovery vs raw, +1.5 pts
[−9.1, +11.6]. Low-pass + recovery vs raw + recovery, +1.8 pts [−7.6, +11.6].
On the training starts, raw + recovery vs raw is −5.0 pts [−15.6, +5.9].

Interpretation:

- **Three of the four cells generalise equally well (49–52%).** Only
  low-pass labels without recovery fall to 34%.
- **The effects interact.** Recovery data repairs the low-pass model
  (+18 pts, section 1) but does nothing measurable for raw labels.
  The jitter in raw labels already shows the policy small off-trajectory
  states and corrections, and gentle recovery adds the same kind of
  variation. Low-pass labels remove it, and recovery data puts it back.
- **Low-pass labels trade generalisation for fit.** They give the best
  training-start success (82%) but cause the most force on new starts.
- **Caveat:** raw-label cells use the 10k checkpoint, low-pass cells 20k.
  For raw labels, 10k beat 20k on every start set in the 16 Oct report.


## 5. Denoising steps and inference latency

![Denoising steps vs latency](figures/denoising_latency.png)

SmolVLA produces each 50-action chunk with flow matching, by default in
10 denoising steps. Fewer steps are faster but give a coarser sample.
`mimicvla.training.evaluate --num-steps N` changes this at inference only;
the model is `lowpass_20k`. Latency comes from `mimicvla.training.latency`
on an idle GPU: 20 real start observations × 3 repeats, with a CUDA
synchronisation around each call. Each call runs the full VLM prefix and
denoising, as at every re-plan.

| Denoising steps | Seeds | 72 dataset starts | 55 excluded starts | Median time to insert | Deep-collision rollouts (72) | Chunk latency, GPU (idle) | Chunk latency, CPU (loaded, upper bound) |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 63% [53, 72] | 30% [19, 41] | 2.50 s | 37% | 37 ms | 161 ms |
| 2 | 5 | 68% [58, 78] | 33% [22, 44] | 2.50 s | 32% | 46 ms | – |
| 3 | 5 | 72% [62, 81] | 30% [19, 41] | 2.52 s | 28% | 54 ms | – |
| 5 | 5 | 76% [68, 84] | 33% [22, 45] | 2.58 s | 24% | 71 ms | – |
| 10 (default) | 5 | 80% [71, 87] | 34% [23, 47] | 2.65 s | 20% | 114 ms | 324 ms |

GPU, idle (RTX 5060 Ti, 16 GB): median 37 ms at 1 step and 114 ms at 10
steps, so each denoising step costs about 8.6 ms on top of about 28 ms for
the VLM prefix. CPU latency (5 observations, measured while four GPU
evaluations shared the CPU, so an upper bound): 161 ms at 1 step and 324 ms
at 10 steps.

Interpretation:

- **10 steps is affordable.** The policy executes 10 actions per chunk at
  20 Hz, which allows 500 ms per chunk. 10 steps uses 23% of that on this GPU
  and still fits on the CPU. If inference ran synchronously, a 114 ms call
  would stall the 50 ms control loop for about two ticks every 10. Running
  inference asynchronously, during the previous chunk, hides it.
- **1 step costs real success.** 1 vs 10 steps: −16.9 pts on dataset starts
  [−24.4, −10.0], and deep collisions rise from 20% to 37%. 5 steps is
  within noise of 10 (−3.6 pts [−7.8, +0.3]) at 62% of the latency.
  Success rises steadily with steps, so 5 is a trade-off, not a free saving.
- Time to insert barely changes (2.50–2.65 s), so fewer steps do not make the
  motion faster, only less accurate.
- On excluded starts every setting sits around 30–34%, the low-pass model's
  generalisation ceiling. Faster sampling does not change where the model
  fails.


## 6. Notes on the evaluation

- **Seeded noise.** `evaluate_policy.py` calls `torch.manual_seed(seed)` once
  per process. The first rollout of a process therefore always draws the same
  noise. This is expected, but it means a single rollout repeated in a fresh
  process will always give the same outcome. Per-start rates from 5 seeds are
  coarse (0, 20, …, 100%). The README videos use seed 1 and print each
  policy's 5-seed record for that start, so a single clip is not mistaken for
  a rate.
- **CPU vs GPU outcomes differ.** Rollouts on the CPU did not reproduce GPU
  outcomes for the same seed. The CPU and CUDA random generators produce
  different noise for the same seed, and kernels differ numerically, so this
  is expected, but it was not investigated further. All success numbers here are GPU rollouts; the CPU was
  used only for latency.
- **Loaded-GPU timing.** `evaluate_policy.py` also records per-rollout
  `median_chunk_ms`. While four evaluations share the GPU, it reads about
  290 ms at 10 steps. Use the idle benchmark above, not those numbers.


## 7. Development history (single rollouts, 14–15 Oct)

These runs used one rollout per start on the 72 dataset starts. With 72
starts one standard error is about ±6 points, and with 8 holdout starts one
rollout moves the rate by 12.5 points, so small differences here are noise.
They are kept because they explain the choices behind the final runs.

### Models

| Run | Data | Train clips | Batch | Steps |
|---|---|---|---|---|
| `no_funnel_baseline_30000` | raw v2 | 72 (no holdout) | 1 | 30k |
| `raw_bs32_20k` | raw v2 | 64 | 32 | 20k |
| `lowpass_bs32_20k` | 2 Hz low-pass tracked poses | 49 | 32 | 20k |
| `recovery_bs32_10k` | raw v2 + DART-style recovery copies (gain 1.0) | 64 (254 episodes) | 32 | 10k |

### Results (all 72 start states, re-plan every 10 steps unless noted)

| Run | Checkpoint | Success | Training starts | Holdout | Deep collisions |
|---|---|---|---|---|---|
| `no_funnel_baseline_30000` (batch 1) | 30k | 23/72 (32%) | trained on all 72 | – | 18 |
| `raw_bs32_20k` | 5k | 29/72 (40%) | 26/64 | 3/8 | 34 |
| `raw_bs32_20k` | 10k | 43/72 (60%) | 39/64 | 4/8 | 1 |
| `raw_bs32_20k` | 20k | 40/72 (56%)* | 35/64 | 5/8 | 3 |
| `lowpass_bs32_20k` | 10k | 57/72 (79%) | 52/64 | 5/8 | 15 |
| `lowpass_bs32_20k` | 15k (stopped at 51) | 46/51 (90%) | 42/46 | 4/5 | 5 |
| `lowpass_bs32_20k` | **20k** | **55/72 (76%)** | 50/64 | 5/8 | 17 |
| `recovery_bs32_10k` | 5k | 17/72 (24%) | 14/64 | 3/8 | 52 |
| `recovery_bs32_10k` | 10k | 18/72 (25%) | 17/64 | 1/8 | 33 |

\* Two separate runs: 35/64 on training starts and 5/8 on the holdout.
"Training starts" here means the 64 v2 training clips. The low-pass model
trained on only 49 of them (see finding 4). The holdout-only runs of
`raw_bs32_20k` scored 5/8 at every checkpoint (5k, 10k, 15k, 20k).

### 1. Batch size was the first-order problem

The first baseline trained with batch size 1 (about 3.7 passes over the data).
On its own training starts it succeeded 23/72 times, with 18 deep collisions.
An open-loop check on training frames showed it had learned only the forward
approach (R² 0.64 on the 10-step sum of dx). Every lateral and rotational
dimension scored below zero, meaning worse than predicting the mean. Batch 32
with a holdout (`raw_bs32_20k`) reached 40–43/72 and 5/8 on the holdout.

### 2. Failures are drift, not near misses

Across every raw-label run, no failed rollout ends near the port: zero failures
finish within 10 mm laterally. Failed rollouts end several centimetres away,
typically tilted 30–50°. Success and failure split early. Either the policy
tracks the approach, or it drifts and keeps drifting, the classic
compounding-error pattern of behaviour cloning on clean demonstrations.

Train and holdout success rates are similar (raw 20k: 35/64 = 55% training
starts against 5/8 holdout), so this is not overfitting to the demonstrations.
The policy cannot reliably reproduce even the motions it trained on.

### 3. Re-planning more often helps a little, and costs collisions

| Checkpoint | Re-plan every | Success | Deep collisions |
|---|---|---|---|
| raw 10k | 10 | 43/72 | 1 |
| raw 10k | 2 | 43/72 | 8 |
| raw 20k | 10 | 40/72 (35/64 + 5/8, separate runs) | 3 |
| raw 20k | 5 | 40/72 | 4 |
| raw 20k | 2 | 49/72 | 8 |
| raw 20k | 2 (seed 1, stopped at 55) | 38/55 | 8 |

Re-planning every 2 steps gave +9 rollouts at 20k but nothing at 10k, and it
consistently raised deep collisions. The likely cause is that consecutive
chunks are independent noise samples, so frequent switching makes motion
jerkier at contact. Ensembling overlapping chunks would avoid that switching.
It is a cheap knob, not a fix for drift.

### 4. Low-pass labels: better fit, unclear generalisation

Marker-tracked orientation is noisy. 72% of the tip's angular-velocity power
lies at or above 5 Hz and only 7% below 2 Hz, while deliberate hand motion
lies below 2–3 Hz. Re-retargeting with a 2 Hz zero-phase Butterworth halves
per-step rotation jitter and keeps each clip's net motion within about 1 mm.

The cost: 17 of 72 clips no longer insert in the 15 × 7 mm port. Most stall on
the rim 0–1 mm deep, 5–12 mm off-centre. Filtering only position or only
orientation breaks different clips, and cutoffs up to 6 Hz do not recover
them. Success through this funnel-less port depends on fine detail of the raw
motion, possibly the small wiggle the jitter adds.

Even so, the low-pass model is the best so far: 55/72 (76%) at 20k and 57/72
(79%) at 10k, against 40–43/72 for raw labels. Split by which clips each model
trained on, though, the gain is almost entirely on clips it trained on.

| Start states (20k checkpoints) | Low-pass | Raw (re-plan 2) | Raw (re-plan 5) |
|---|---|---|---|
| 49 clips the low-pass model trained on | 41/49 | 35/49 | 29/49 |
| 15 clips the filter rejected (raw trained on them, low-pass did not) | 9/15 | 9/15 | 7/15 |
| 8 holdout clips (neither trained on them) | 5/8 | 5/8 | 4/8 |

So smoother labels make the demonstrations much easier to reproduce, and the
model fits its own clips better. On clips the low-pass model never saw, it
matches the raw model, even on the 15 clips the raw model *did* train on. With
23 unseen starts, a generalisation gain cannot be claimed either way.

Its failures also look different. They end about 15 cm away, tilted only
14–17° (raw: 28–50°), but with higher peak force (about 500 N), and deep
collisions rose to 15–17 rollouts. Rotation drift is mostly gone; failures now
look like lateral misses that end pushing on the port face.

### 5. Recovery data v1 failed, for an identifiable reason

`mimicvla.dataset.recovery` re-executes each training clip 3 extra
times. Each copy adds an Ornstein–Uhlenbeck offset (about 3 mm and 2° RMS,
fading to zero within 5 mm of the mouth) to the executed command, and labels
each frame with the clean command back to the human target. Copy 0 reproduces
the v2 labels to 2.5e-6, and 254 of 256 copies pass the v2 gate.

It made the policy much worse: 17/72 (24%) at 5k and 18/72 (25%) at 10k,
against 29/72 and 43/72 for raw labels at the same steps. Holdout success fell
to 1/8 at 10k. Failed rollouts end about 34 cm away and about 89° off.

The labels removed the whole offset in one step (capped at 3 mm and 2°). With
a 2° RMS rotation offset the cap is hit often, so the rotation label std rose
from 7.3 to 12.7 mrad per step. The policy learned large, abrupt rotation
commands. In closed loop these overshoot and build into the 25 cm, 70°
excursions seen in the results. There is a second issue: the labels depend on
the controller's internal target, which the policy cannot observe, because
the arm's actual pose lags the target.

The fix is a partial correction per step. `--correction-gain` (default 1.0,
which is v1) sets the fraction of the offset each label removes. On 5 sample
clips, gain 0.15 brings label spread back to human scale: 7.0–7.7 mrad and
1.0–1.2 mm per step, against 7.3 mrad and 0.8–1.3 mm in v2. The robot then
returns over about 0.3 s, with the same noise and the same gate pass rate.
It was trained afterwards; see the follow-up below and section 4.

### Follow-up (15 Oct): gentle recovery labels and DSRL

All runs on the 72 start states, one rollout each, re-plan every 10 steps.

| Run | Success | Training starts | Holdout | Deep collisions | Median peak force |
|---|---|---|---|---|---|
| `lowpass_bs32_20k` 20k (reference) | 55/72 (76%) | 50/64 | 5/8 | 17 | 128 N |
| `lowpass_recovery_bs32_20k` 10k | 46/72 (64%) | 41/64 | 5/8 | 26 | 176 N |
| `lowpass_recovery_bs32_20k` 20k | 53/72 (74%) | 49/64 | 4/8 | 19 | 140 N |
| `recovery_v2_bs32_10k` 10k (raw + gentle recovery) | 44/72 (61%) | 40/64 | 4/8 | 12 | 135 N |
| DSRL on low-pass 20k, base noise N(0, I) | 55/72 (76%) | 50/64 | 5/8 | 17 | 123 N |
| DSRL on low-pass 20k, fixed zero noise | 58/72 (81%) | 53/64 | 5/8 | 14 | – |
| DSRL on low-pass 20k, learned steering (30k decisions) | 59/72 (82%) | 54/64 | 5/8 | 13 | 131 N |

- **Gentle recovery labels** (`--correction-gain 0.15`) remove the damage of v1
  (18/72 → 44/72 on raw labels), but add nothing over training without
  recovery, on raw or low-pass labels.
- **DSRL** (`mimicvla.training.dsrl`) keeps SmolVLA frozen and trains SAC
  over the initial flow-matching noise of the 10 executed steps (60 dims,
  bounded to ±2), on training starts only. Its training success stayed at
  about 80% throughout. The learned steering scores 59/72 deterministically,
  but a fixed all-zero noise scores 58/72, so the gain over random noise is
  explained by removing sampling noise, not by anything the RL learned
  (against zero noise: 5 starts gained, 4 lost; holdout unchanged).
- **Single-rollout comparisons are noisy.** Across these five evaluations of
  low-pass-based policies, 40 starts always succeed, 6 always fail (3 of them
  holdout) and 26 flip between runs. Differences of ±4/72 are within noise;
  future comparisons need several seeds per start.


## Videos

`mimicvla.training.videos` builds the clips in `documentation/videos/`
(full-quality `.mp4` and a compact `.gif`). Each shows the iPhone recording
with the ArUco marker axes and projected socket and USB tip, the Panda
replaying the retargeted motion, and SmolVLA rolled out from the same start
(seed 1). Every simulation panel is a close three-quarter view with the
policy's front and wrist cameras below. The outcome appears on the held final
frame, with the start's result over the 5 evaluation seeds.

| Video | Start | What it shows |
|---|---|---|
| `unseen_start_222751` | excluded | the replay misses; the raw policy inserts in 2.7 s (5/5 seeds) |
| `unseen_start_222751_lowpass_vs_raw` | excluded | same start: low-pass fails (0/5), raw inserts (5/5) |
| `unseen_start_182731_fail` | excluded | the raw policy fails (0/5) and ends about 5 cm off the port |
| `holdout_dual_marker_223447` | holdout | two-marker recording; raw policy inserts (5/5) |
| `holdout_single_marker_194743` | holdout | single-marker recording; raw policy inserts (4/5) |

## Reproduce

`scripts/reproduce_results.sh` lists the dataset exports, training runs,
evaluations and summaries in order. Single steps:

```bash
# train (the 8 holdout clips are excluded automatically)
bash scripts/train.sh RUN STEPS [DATASET_ID] [BATCH]
# evaluate one seed on a start set
MUJOCO_GL=egl .venv/bin/python -m mimicvla.training.evaluate CKPT \
  --dataset-dir data/mimicvla-panda-usb-15x7/eval_starts/excluded55 --split all \
  --n-action-steps 10 --no-video --seed 0 --output-dir OUT
# tables, intervals and figures
.venv/bin/python -m mimicvla.training.report
```
