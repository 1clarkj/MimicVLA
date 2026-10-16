# MimicVLA

The goal of this project is to teach a robot arm a precise manipulation task
from a few phone videos of myself doing it. I recorded 134 clips of plugging a
USB stick into a computer's USB port with a handheld iPhone, turned them into
Franka Panda demonstrations in MuJoCo, and fine-tuned the SmolVLA
vision-language-action model on them. The resulting policy inserts the plug
about half the time from start poses it never trained on.

Results, engineering choices and what did not work are in
**[RESULTS.md](RESULTS.md)**.

![iPhone recording, its Panda replay, and the trained policy from an unseen start](documentation/videos/unseen_start_222751.gif)

*Left to right: the iPhone recording with marker tracking, the Panda replaying
that motion in MuJoCo, and the trained policy starting from the same pose. This
recording was never used for training. Each simulation panel shows a close
view with the policy's front and wrist cameras below it.*

## How it works

1. **Track.** ArUco markers on the plug and next to the port give the plug's
   pose relative to the port in every video frame.
2. **Replay.** That trajectory is retargeted to the Panda's hand and replayed
   in MuJoCo. Two simulated cameras record images, joint states and actions,
   which become a LeRobot dataset.
3. **Train.** SmolVLA is fine-tuned on those demonstrations.
4. **Evaluate.** The policy runs closed loop in MuJoCo from start poses taken
   from recordings it never trained on.

The policy sees only the two simulated cameras, the robot state and the
instruction "Insert the USB stick into the USB port." The phone video is used
only to measure the motion.

## Repository

```
src/mimicvla/
  vision/        camera calibration and ArUco marker tracking
  retarget.py    plug trajectory → Panda end-effector commands
  sim/           MuJoCo scene: Panda, USB plug, wall port, two cameras, IK controller
  dataset/
    tracking/    track every recording, fit the marker-to-plug offset per session
    replay/      replay recordings on the Panda (real and widened port)
    review/      review queues and the browser dashboard for accepting clips
    selection/   port-size sweep, replay audit, LeRobot export
    variants/    low-pass, recovery and mirrored label variants
    publish/     Hugging Face upload and download
  training/      closed-loop evaluation, DSRL, latency benchmark, result tables, videos
  cli.py         the `mimicvla` command
scripts/         train.sh, process_recordings.sh, reproduce_results.sh, view_sim.sh
calibration/     printable ChArUco and ArUco marker sheets
documentation/   figures and videos used in RESULTS.md
tests/           unit and integration tests
```

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). Training needs a GPU with
16 GB (an RTX 5060 Ti was used); everything else runs on a CPU.

```bash
git clone https://github.com/1clarkj/MimicVLA && cd MimicVLA
uv sync --group train
export MUJOCO_GL=egl                                 # headless rendering
export MENAGERIE_CACHE_DIR=outputs/menagerie_cache   # Panda model, downloaded on first use
uv run pytest -q
```

## Quick start with the published data

```bash
uv run mimicvla download --policy        # training set, start states and trained policy → data/
uv run mimicvla sim-demo                 # scripted insertion → outputs/sim_demo.mp4
bash scripts/view_sim.sh                 # interactive viewer

# Evaluate the trained policy from the 55 start poses it never trained on
uv run mimicvla evaluate data/mimicvla-smolvla-usb --seeds 0 1 2 3 4

# Train your own policy (about 1 h for 10k steps on one GPU)
bash scripts/train.sh my_run 10000
uv run mimicvla evaluate outputs/train/my_run/checkpoints/last/pretrained_model
```

`mimicvla evaluate --starts holdout` uses the 8 held-out clips instead, and
`--video` saves a video of every rollout. `bash scripts/reproduce_results.sh`
reruns every experiment in RESULTS.md.

## Recording your own demonstrations

Print the two sheets in `calibration/`. Tape a 20 mm marker 0 to the USB stick
and the 50 mm marker 1 next to the port. The marker-to-plug-tip and
marker-to-port offsets of my setup are in
`src/mimicvla/vision/target_frames.py`; measure and change them for yours.

**1. Calibrate the phone.** Film the flat ChArUco sheet for about 30 seconds.
Move the phone around it: tilt it about 30° left, right, up and down, film it
from near and far, and move the board into each corner of the image. Use the
same lens, zoom and resolution as for the demonstrations, and turn off video
stabilisation and automatic lens switching.

```bash
uv run mimicvla calibrate charuco.mov    # → outputs/iphone_calibration.json
```

It needs at least 6 usable views and warns when the board did not cover
enough of the image.

**2. Record and track a demonstration.** Keep both markers in view from a few
centimetres before the port until the plug is seated.

```bash
uv run mimicvla track demo.mov           # → outputs/tracking/demo/trajectory.jsonl
uv run mimicvla replay outputs/tracking/demo/trajectory.jsonl   # → outputs/tracking/demo/replay/replay.mp4
```

**3. Build a training set from many recordings.** Put single-marker videos in
`video_data/20mm_marker/` and two-marker videos in
`video_data/double_marker_original_mount/`, then:

```bash
bash scripts/process_recordings.sh track    # track, recalibrate and replay every clip
uv run python -m mimicvla.dataset.review    # accept or reject each clip in the browser
bash scripts/process_recordings.sh select   # → outputs/datasets/usb_15x7_no_funnel_full_v2
bash scripts/train.sh my_run 10000 outputs/datasets/usb_15x7_no_funnel_full_v2
```

## Data

- [`<HF_USER>/mimicvla-panda-usb-15x7`](https://huggingface.co/datasets/<HF_USER>/mimicvla-panda-usb-15x7):
  the 72-episode LeRobot training set and the evaluation start states.
- [`<HF_USER>/mimicvla-human-usb-raw`](https://huggingface.co/datasets/<HF_USER>/mimicvla-human-usb-raw):
  the 134 iPhone recordings with per-frame tracking and per-clip labels
  (`mimicvla download --raw`, 6.8 GB).
- [`<HF_USER>/mimicvla-smolvla-usb`](https://huggingface.co/<HF_USER>/mimicvla-smolvla-usb):
  the trained policy (raw labels, 10k steps).

## License

MIT. The Panda model comes from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) under
its own license.
