# Experiments

Everything behind [RESULTS.md](../RESULTS.md): the full tables of every
policy, the development history that led to the final protocol, and the
details of the data pipeline. Policy numbers are closed-loop MuJoCo
rollouts, not demonstration replays. Recording and tracking are covered in
[data_collection.md](data_collection.md).

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
  (`1clarkj/mimicvla-panda-usb`) used for training.
- **Holdout (8)**: clips of the same dataset that no model trained on
  (`mimicvla.dataset.splits`, seed 2026, stratified by recording session).
  Too few to rank models.
- **Excluded starts (55)**: aligned start poses from recordings that never
  entered any training set (32 accepted in review but failed the 15 × 7 mm
  replay gate, 7 rejected or held in review, 16 never reviewed). Only the
  start pose is used. **This is the main test of generalisation.**

Sections 1–5 use 5 seeds per start. A start's success is its mean over seeds;
intervals are 95% bootstraps over start states, and differences are paired
over the same starts. Section 6 records the earlier single-rollout
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

Every policy ran 5 seeds from each start. The excluded starts are
`eval_starts/excluded55` in the published dataset.

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

What these differences mean is discussed in [RESULTS.md](../RESULTS.md),
steps 2–4.

## 2. Data scaling

Same recipe as `raw_10k` (raw labels, batch 32, 10k steps, learning-rate
schedule of the 20k run so the 64-demo point is the same checkpoint).
`mimicvla.dataset.splits` draws nested subsets of the 64 training clips
(seed 2026): 16 ⊂ 32 ⊂ 64.

| Demos | Policy | Trained starts | Excluded starts (55) | Deep-collision rollouts (55) | Median peak force (55) |
|---|---|---|---|---|---|
| 16 | `raw_n16_10k` | 48% [31, 64] | 16% [9, 24] | 75% | 610 N |
| 32 | `raw_n32_10k` | 45% [34, 56] | 24% [16, 32] | 21% | 285 N |
| 64 | `raw_10k` | 60% [52, 68] | 49% [39, 59] | 4% | 218 N |
| 110 (64 + 46 mirrored) | `raw_mirror_10k` | 47% [40, 54] | 40% [32, 49] | 17% | 222 N |

Paired differences on the 55 excluded starts:

- 64 vs 32 demos: **+25.1 pts** [+14.9, +35.6].
- 64 vs 16 demos: **+33.1 pts** [+21.8, +44.0].
- 32 vs 16 demos: +8.0 pts [−0.4, +16.7].

Interpretation:

- **More human demonstrations help, and the curve has not flattened.** The
  gain from 32 to 64 is larger than from 16 to 32. Success on unseen starts
  roughly doubles and deep collisions drop from 75% to 4% of rollouts. The
  gap between trained and unseen starts shrinks from about 30 points at 16
  demos to about 10 at 64.
- **At 16 and 32 demos the policy memorises.** It reproduces about half of
  its own starts but only about a fifth of new ones.
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

- `mimicvla build-dataset --mirror` → `usb_15x7_mirror`: 120 episodes,
  13,806 frames. Of 72 mirrored clips, **48 pass the same replay gate** as
  the originals. Almost all 24 rejects stall on the rim at about 0 mm depth,
  the same failure as low-pass breakage; the human's fine corrections are
  probably not symmetric about the opening.
- Mirrors of holdout clips stay in the holdout, so 110 episodes train
  (64 originals + 46 mirrors).
- Applied to raw labels, because they give the best policy without recovery
  data and keep all 72 clips (low-pass labels break 17).

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

A 20k-step mirror run separates these and is the next run to train. If the
mirrors were useful, they should already have helped at 10k steps, so the
second reason is the more likely one, but only the longer run can show it.

**Lighting augmentation was skipped.** The simulator renders with fixed
lighting, so evaluation in sim cannot show a benefit. It belongs to
sim-to-real transfer, which this project does not test.

## 4. Labels × recovery data (2 × 2)

![Labels × recovery](figures/labels_x_recovery.png)

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
  For raw labels, 10k beat 20k on every start set (section 1).

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

## 6. Development history (single rollouts)

These runs used one rollout per start on the 72 dataset starts. With 72
starts one standard error is about ±6 points, and with 8 holdout starts one
rollout moves the rate by 12.5 points, so small differences here are noise.
They are kept because they explain the choices behind the final runs.

### Models

| Run | Data | Train clips | Batch | Steps |
|---|---|---|---|---|
| `no_funnel_baseline_30000` | raw | 72 (no holdout) | 1 | 30k |
| `raw_bs32_20k` | raw | 64 | 32 | 20k |
| `lowpass_bs32_20k` | 2 Hz low-pass tracked poses | 49 | 32 | 20k |
| `recovery_bs32_10k` | raw + DART-style recovery copies (gain 1.0) | 64 (254 episodes) | 32 | 10k |

### Results (all 72 start states, re-plan every 10 steps unless noted)

| Run | Checkpoint | Success | Training starts | Holdout | Deep collisions |
|---|---|---|---|---|---|
| `no_funnel_baseline_30000` (batch 1) | 30k | 23/72 (32%) | trained on all 72 | – | 18 |
| `raw_bs32_20k` | 5k | 29/72 (40%) | 26/64 | 3/8 | 34 |
| `raw_bs32_20k` | 10k | 43/72 (60%) | 39/64 | 4/8 | 1 |
| `raw_bs32_20k` | 20k | 40/72 (56%)* | 35/64 | 5/8 | 3 |
| `lowpass_bs32_20k` | 10k | 57/72 (79%) | 52/64 | 5/8 | 15 |
| `lowpass_bs32_20k` | **20k** | **55/72 (76%)** | 50/64 | 5/8 | 17 |
| `recovery_bs32_10k` | 5k | 17/72 (24%) | 14/64 | 3/8 | 52 |
| `recovery_bs32_10k` | 10k | 18/72 (25%) | 17/64 | 1/8 | 33 |

\* Two separate runs: 35/64 on training starts and 5/8 on the holdout.
"Training starts" here means the 64 training clips. The low-pass model
trained on only 49 of them (see 4 below). The holdout-only runs of
`raw_bs32_20k` scored 5/8 at every checkpoint (5k, 10k, 15k, 20k).

### 1. Batch size was the first-order problem

The first baseline trained with batch size 1 (about 3.7 passes over the
data), because it was not yet clear how much GPU memory fine-tuning would
need; batch 32 turned out to fit in 16 GB.
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
them. Success through this tight port depends on fine detail of the raw
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
the raw labels to 2.5e-6, and 254 of 256 copies pass the replay gate.

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
1.0–1.2 mm per step, against 7.3 mrad and 0.8–1.3 mm for the raw labels. The robot then
returns over about 0.3 s, with the same noise and the same gate pass rate.
It was trained afterwards; see the follow-up below and section 4.

### Follow-up: gentle recovery labels and DSRL

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

## Data pipeline details

Recording and tracking problems are in [data_collection.md](data_collection.md).

### Retargeting and simulation

`mimicvla.retarget` turns the tracked `T_socket_usb_tip` segment into robot
commands: it resamples to 20 Hz, removes isolated pose outliers, lightly
smooths translation while keeping the final depth, and limits each Cartesian
target step to 3 mm and 2°. The targets are placed at the simulated socket,
and a short bridge from the reset pose to the first tracked pose is added
(the "alignment" that evaluation also replays to reach each start state).
Mink differential IK turns end-effector targets into joint targets, and the
Panda's position actuators track them.

| Item | Definition (`src/mimicvla/sim/env.py`) |
|---|---|
| Robot | MuJoCo Menagerie Panda; the USB plug (12 × 4.5 mm) is rigidly attached to the hand |
| Action | 6 floats at 20 Hz: world-frame `dx, dy, dz` (m) and a rotation vector (rad) applied to the end-effector target |
| State | 9 floats: seven joint angles (rad), two finger positions (m) |
| Cameras | `camera1` fixed front, `camera2` on the wrist; 256 × 256 RGB |
| Port | rectangular opening in a flush wall, 18 mm deep; 15 × 7 mm unless noted |
| Success | plug-tip centre inside the opening and at least 12 mm deep |
| Deep collision | more than 1 mm penetration into the port walls (wall contact itself is allowed) |

### Selection

1. **Review: 104 of 134 clips accepted.** Each clip was reviewed next to its
   replay into an 18 × 10 mm port; 5 were rejected, 2 held, 23 never
   reviewed.
2. **Port sweep: 75 clips.** The 103 accepted clips available at the time
   were replayed into 46 openings. In the 15 × 7 mm port, 75 reach 7 mm tip
   depth during the human motion without a deep collision or joint-limit
   fault.
3. **Full replay: 72 clips.** Checked to the end of the replay, two have late
   deep collisions and one ends shallower than 12 mm. The remaining 72 (36
   single-marker, 36 two-marker) are the training set; `mimicvla
   build-dataset` applies steps 2 and 3 in one pass.

Port sweep (a clip counts if it passes 7 mm deep during the human motion with
no deep collision or joint-limit fault up to that point):

| Opening | Clips | Area | Clips over 100 N peak |
|---|---:|---:|---:|
| 18 × 10 mm (review port) | 103 | 180 mm² | 51 |
| 16 × 8 mm | 89 | 128 mm² | 49 |
| 15 × 8 mm | 85 | 120 mm² | 50 |
| **15 × 7 mm (chosen)** | 75 | 105 mm² | 44 |
| 14 × 7 mm | 70 | 98 mm² | 47 |
| 17.4 × 5.1 mm | 70 | 89 mm² | 62 |
| 13.4 × 5.9 mm (real port) | 44 | 79 mm² | 34 |

17.4 × 5.1 mm is the smallest area with 70 clips, but it leaves 0.3 mm over
the plug height and 62 of its clips exceed 100 N. 14 × 7 mm has no margin
for later rejections. An earlier design with an entry funnel let more
replays insert but pushed the plug hard against the funnel, so it was
dropped.

Clips over 100 N peak force were kept; 100 N was a triage screen, not a
limit. Each episode's peak force is in `capture_manifest.json`.

The 8 held-out clips (`mimicvla.dataset.splits`, seed 2026) are stratified
by session: 4 of 36 single-marker, 3 of 32 two-marker, 1 of 4 first-mount.
The split is keyed by clip name, so every derived dataset holds out the same
demonstrations, mirrored copies included.

### Dataset variants

All are built from the same 72 clips by `scripts/reproduce_results.sh data`
and keep only episodes that pass the same replay gate.

| Dataset | Built with | Episodes | Frames | Change |
|---|---|---:|---:|---|
| `usb_15x7` (published as `mimicvla-panda-usb`) | `build-dataset` | 72 | 8,089 | raw tracked poses |
| `usb_15x7_lowpass2hz` | `build-dataset --lowpass-hz 2` | 55 | 4,768 | zero-phase 2nd-order Butterworth at 2 Hz on tip position and orientation |
| `usb_15x7_recovery` | `dataset.recovery --correction-gain 0.15` | 254 | 28,821 | 1 clean + 3 perturbed copies per training clip |
| `usb_15x7_lowpass2hz_recovery` | same, from the low-pass set | 185 | – | the same on low-pass labels |
| `usb_15x7_mirror` | `build-dataset --mirror` | 120 | 13,806 | 72 originals + 48 mirrored copies |

**Low-pass.** 72% of the tracked tip's angular-velocity power is at or above
5 Hz and 7% below 2 Hz, while deliberate hand motion lies below 2–3 Hz.
Filtering roughly halves per-step rotation jitter and changes net motion by
about 1 mm, but 17 clips then stall on the rim (0–1 mm deep, 5–12 mm off
centre). This leaves 55 episodes, 6 of them held out.

**Recovery.** Each training clip is re-executed four times: once clean and
three times with a smooth Ornstein–Uhlenbeck offset (about 3 mm and 2° RMS)
added to the executed command, fading to zero within 5 mm of the port. Each
frame is labelled with the command that moves the target back towards the
human trajectory, removing `--correction-gain` of the offset per step (1.0
in version 1, 0.15 in version 2). Held-out clips are not used. 254 of 256
copies pass the gate.

**Mirror.** Each tracked tip trajectory is reflected across the port's
vertical mid-plane (`mirror_socket_x` in `src/mimicvla/retarget.py`), then
retargeted and rendered like an original (section 3).

## Reproduce

`scripts/reproduce_results.sh` rebuilds the dataset variants, trains every
run and evaluates it, in order. Single steps:

```bash
bash scripts/train.sh RUN STEPS [DATASET] [BATCH]       # holdout clips excluded automatically
uv run mimicvla evaluate CKPT --seeds 0 1 2 3 4          # 55 excluded starts
uv run python -m mimicvla.training.report                # tables, intervals, figures
```

Lower-level options (`--n-action-steps`, `--num-steps`) are on
`python -m mimicvla.training.evaluate --help`.
