# USB insertion evidence clips

These compact MP4s preserve examples for the project README and presentation.
They are **video only, with no audio**. Source videos and complete trajectory
reports remain in `video_data/` and `outputs/`; `manifest.json` records the
source of each copied replay and the frames used for the annotated excerpts.

| Stage | Clip | What it demonstrates |
| --- | --- | --- |
| Human recording and missing marker | [single_marker_tracking_gap.mp4](single_marker_tracking_gap.mp4) | Marker 0 disappears in frames 34–51 of `10102026_194658.mov`. The overlay shows the resulting missing USB pose. The pipeline does not fabricate that 18-frame span. |
| Original pose estimate | [single_marker_before_pose_prior.mp4](single_marker_before_pose_prior.mp4) | An ambiguous planar marker pose caused large target changes. MuJoCo replay needed 360 controller steps, violated joint limits, and had 130 deep-collision steps. The IK controller was following a faulty estimated target; this is not evidence of an intrinsic IK defect. |
| Pose-prior correction | [single_marker_after_pose_prior.mp4](single_marker_after_pose_prior.mp4) | The same source clip with temporal/task pose selection needed 95 steps, respected joint limits, and had zero deep-collision steps. Forty wall-contact steps remain. It passes the corrected 12 mm insertion criterion; wall contact is acceptable. |
| Masking and rejection | [single_marker_tracking_gap.mp4](single_marker_tracking_gap.mp4) | Long gaps are left missing; interpolation is limited to four frames under motion limits. Four early 12 mm marker clips have no replayable span. The raw dataset's `quality_review.csv` and `videos/metadata.jsonl` retain explicit exclusion reasons. |
| Second marker during occlusion | [dual_marker_visibility.mp4](dual_marker_visibility.mp4) | In `10112026_194713.mov`, marker 0 disappears around frames 72–94 while marker 10 recovers many poses. Marker-0-only tracking has no qualifying replay span; dual tracking does. |
| One marker replay, restored mount | [new_batch_marker0_only.mp4](new_batch_marker0_only.mp4) | `10112026_194839.mov` replayed using marker 0 alone. |
| Two marker replay, restored mount | [new_batch_two_markers.mp4](new_batch_two_markers.mp4) | The same clip replayed with marker 10 primary and marker 0 fallback. Dual tracking covers 1,001/1,056 frames across the seven-video batch, versus 705/1,056 with marker 0 only. The replay still contacts the simulated wall. |
| Depth-and-entry success | [dual_194811_depth_entry.mp4](dual_194811_depth_entry.mp4) | `10112026_194811.mov` reaches 17.96 mm maximum tip depth. Its tip enters the modeled opening past 12 mm, so the current task rule counts it as a success. The earlier strict angle/contact rule rejected it. |
| Corrected simulation target | [scripted_12mm_success.mp4](scripted_12mm_success.mp4) | A scripted Panda path reaches 14.7 mm tip depth with no wall contact under the corrected 12 mm insertion criterion. |

The current 12 mm rescore is preserved here as
[insertion_depth_entry_12mm_summary.json](insertion_depth_entry_12mm_summary.json)
and [insertion_depth_entry_12mm_results.csv](insertion_depth_entry_12mm_results.csv);
the full report is in `outputs/evaluation_depth_entry_12mm/`. It uses the
saved *actual* MuJoCo poses. The task rule requires the tip centre inside the
rectangular port opening at 12–18 mm depth. It counts 10/62 original
single-marker videos, 2/9 first-batch dual-marker videos, and 3/7 restored-mount
dual-marker videos. The previous stricter scores are preserved in
[insertion_strict_12mm_summary.json](insertion_strict_12mm_summary.json) and
[insertion_strict_12mm_results.csv](insertion_strict_12mm_results.csv).
Orientation error, wall contact, deep simulated penetration, and robot replay
quality remain available for manual review; they do not change task success.
These numbers assess replay under this simulation model; they do not determine
whether the human inserted the plug successfully.

The current two-marker fit has about 2.5 mm median position residual on
co-visible frames, and individual clips reach 5–17 mm marker disagreement at
the 95th percentile. That matters for a millimetre-scale USB socket. The two
markers improve visibility, while calibration and contact geometry remain
open problems.

Regenerate the curated clips from the source videos and replay outputs with:

```bash
.venv/bin/python scripts/build_documentation_examples.py
```
