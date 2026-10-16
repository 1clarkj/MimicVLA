# Data collection

How the 134 phone recordings were made, what went wrong, and what I changed
along the way. The summary is in [RESULTS.md](../RESULTS.md); how to record
your own data is in the [README](../README.md#option-b-record-your-own-demonstrations).

## Rig

- **Phone:** iPhone, 4K (3840 × 2160) at about 24 fps, handheld. The phone
  is allowed to move: the plug pose is measured relative to the port marker,
  so the camera pose cancels.
- **Camera app:** [Filmic Pro](https://www.filmicpro.com/) (free tier),
  with focus, exposure and lens locked. The stock Camera app refocuses,
  re-exposes, switches lenses and stabilises (crops) the image on its own.
  Each of these changes the camera intrinsics in the middle of a clip,
  so the calibration no longer holds.
- **Calibration:** a 5 × 7 ChArUco board (30 mm squares) filmed with the same
  settings: 33 usable views, 0.64 px RMS reprojection error.
- **Markers** (`DICT_4X4_50`, printed from `calibration/`):
  - port marker id 1, 50 mm, taped flat next to the port;
  - plug marker id 0, 20 mm (an early 12 mm version was too small);
  - from the second half of the recordings, a second plug marker, id 10,
    22 mm, on a different face of the plug.
- **Task:** one person inserting a USB-A stick into a computer's USB port, from
  varied approach positions and angles, correcting before entry.

## Recording sessions

| Session | Clips | Plug markers | Frames with a plug pose (median) | Longest tracking gap (median) | Accepted in review | In the training set |
|---|---:|---|---:|---:|---:|---:|
| `single_12mm` | 4 | id 0, 12 mm | 11% | 71 frames | 0 | 0 |
| `single_20mm` | 58 | id 0, 20 mm | 70% | 19 frames | 47 | 36 |
| `dual_minus3` | 9 | id 0 + id 10 (first mount) | 96% | 0 frames | 7 | 4 |
| `dual_original` | 63 | id 0 + id 10 | 90% | 7 frames | 50 | 32 |

Each session has its own setup file (`setups/` in the
[raw dataset](https://huggingface.co/datasets/1clarkj/mimicvla-human-usb-raw)),
because the marker-to-plug-tip offset differed between sessions (see below).

## Problems and fixes

### 1. The plug marker disappears during insertion

With one marker, the hand covers it or motion blur hides it exactly when it
matters: during the last centimetres before entry. In the single-marker
session the plug pose was missing in 30% of frames, and the median clip had a
19-frame (0.8 s) gap. The tracker does not invent poses: gaps of up to four
frames are interpolated, and longer gaps split the clip. A clip is replayable
only if one continuous segment covers the approach and the insertion.

**Fix: a second plug marker.** Marker 10 sits on another face of the plug,
so at least one marker is usually visible. Its transform to marker 0 is
fitted from frames where both are visible (`mimicvla fit-marker-mount`). On
the first seven test clips, the plug pose was available in 1,001 of 1,056
frames with both markers, against 705 with marker 0 alone, and the median
longest gap fell from 19 to 7 frames over the whole session. **The second
marker is optional in the code, but I highly recommend it.**

It improved coverage, not accuracy. The share of clips that ended up in the
training set did not rise (36/58 single-marker, 36/72 dual), because most
rejections come from the offset error below, which a second marker does not
fix. Even after fitting, the two markers disagree by 5–17 mm (95th percentile)
in some clips, which hints at small motions of the printed mounts.

### 2. Flipping marker orientation

A flat marker seen at a shallow angle has two pose solutions that look almost
the same in the image. The early tracker sometimes jumped between them, which
produced large false rotations. Replayed on the robot, one such clip needed
360 controller steps, violated joint limits and spent 130 steps deep inside
the port walls. The tracker now picks the solution that is consistent with
the previous frame and with the known marker mount. The same clip then
replays in 95 steps without a deep collision.

### 3. The marker-to-tip offset drifts between sessions

The tip's position relative to the plug marker was measured with a ruler.
Every clip ends with the plug physically inserted, so the tracked tip should
end inside the port, yet it ended 3.5–7.5 mm off-centre, by a different amount in
each session. Either a marker shifted on its mount or the tracking has a bias;
the data cannot tell which. I fitted an effective offset per session from the
frames where the plug is inside the port:

| Session / period | Clips | x correction | z correction | Held-out radial error (90th pct) |
|---|---:|---:|---:|---:|
| single 20 mm, early | 5 | −3.6 mm | −0.5 mm | 1.8 mm |
| single 20 mm, later | 52 | −6.2 mm | −0.7 mm | 1.9 mm |
| dual, first mount | 9 | −6.3 mm | −0.4 mm | 1.7 mm |
| dual, periods 1–4 | 5–42 | −3.8 to −7.3 mm | −1.2 to +2.4 mm | 1.0–3.2 mm |

The fit uses the same clips it corrects, so it is not an independent check.
The remaining 1–3 mm error is still larger than the real port's clearance
(0.7 mm per side), which is why the simulated port is widened to 15 × 7 mm
(see [RESULTS.md](../RESULTS.md#the-port-is-widened)).

### 4. Review

I reviewed every clip next to its robot replay (104 of 134 accepted;
`accepted_clips.txt` in the raw dataset). A clip was rejected mostly when the
tracking visibly disagreed with the video or the replay did something the
human did not. Of the 104, 72 pass the replay gate in the 15 × 7 mm port and
form the training set. The start poses of the other 32, plus 7 rejected and
16 unreviewed clips, are the 55 unseen evaluation starts.

## What I would do differently

- **Use two plug markers from the first clip**, on a rigid printed mount
  instead of tape, so the offset cannot drift.
- **Measure the transforms instead of using a ruler:** record a few seconds
  of the plug fully inserted at the start of each session and fit the offset
  from it, rather than correcting afterwards.
- **Move slowly in the last 2 cm.** Fast motion caused most lost
  detections.
- **Record many more clips.** Data was the strongest lever on the policy
  (see RESULTS.md, step 6); each clip takes only seconds to film.
