# SPEC

Eye-tracking analysis for the VLAT line/area annotation study (reVISit + Pupil Core).

## Data per participant (`recordings/<pid>/`)

- `<NNN>/` — Pupil Core recording (Pupil Capture 3.5.7), with Pupil Player exports in `exports/`.
- `VLAT_*_<participant uuid>.json` — reVISit participant export. Timestamps are unix ms.
- `<pid>_screen.mov` — tablet screen recording (ReplayKit, iPad as a Sidecar display of the Mac).

Study order: introduction, then three chart-type blocks (stackarea, area, line; latin-square order).
Each trial is a 5 s fixation cross followed by a chart with a question and a drawing canvas.
From p2 on, each chart is followed by a NASA-TLX slide (`tlx-<chart>-<task>`), before the next fixation cross.
Each block ends with a `validation-break`, during which Pupil Capture runs a screen-marker validation.

## Clocks (`pupil_analysis/timeline.py`)

| clock | origin | conversion |
|---|---|---|
| unix ms | reVISit `Date.now()` | — |
| Pupil s | Pupil Capture synced clock | `unix/1000 + (start_time_synced_s - start_time_system_s)` from `info.player.json`. Valid because the browser runs on the same Mac (Sidecar). |
| world s | first world-video frame (Pupil Player timeline) | `pupil - world_timestamps[0]`; `world_frame` = Player frame index |
| screen s | screen-recording t=0 | `pupil - screen_zero`. `screen_zero` is measured, because the file's `creation_time` is the save time and only has 1 s precision |

Alignment is checked against the videos. Chart slides show a white panel and fixation slides are grey,
so every chart on/off transition from reVISit should show up as a brightness edge in each video.
`fit_offset` matches the edges and reports per-transition residuals in `<pid>_sync_check.csv`.

World-video threshold:
- The world-video white-fraction threshold is measured per participant (`transition_threshold`), because
  camera exposure differs.
- It is the midpoint of the median fixation-slide level (1.5–0.5 s before chart onsets) and chart-slide level
  (0.5–1.5 s after).
  - p1: 0.094 / 0.128 → 0.111. The old fixed value was 0.11. Refitting p1 with the measured threshold moves the
    lag from 97.7 to 102.4 ms, still 30/30 matched. p1's outputs were made with the fixed 0.11 and have not been rerun.
  - p2: 0.002 / 0.043 → 0.022. The fixed 0.11 found no edges at all.
- The screen-recording threshold stays at 0.2: fixation slides are ~0 and charts ≥ 0.38.

p1 results: all 30/30 transitions matched in each video.
- World video lags reVISit by a constant 98 ms (display + camera latency), max residual 48 ms.
- Screen-recording residuals are at most 27 ms.

p2 results: 30/30 matched in each video.
- World lag is 115 ms, max residual 73 ms.
- Screen-recording max residual is 65 ms.

Event times are **not** shifted by the 98 ms lag. They are the times reVISit issued the stimulus.

## Output: `<pid>_event_timeline.csv`

Built by `uv run python scripts/event_timeline.py recordings/<pid>`. There is one row per event:
- `fixation_cross`: cross shown → hidden.
- `graph`: chart shown → answer submitted. Includes `answer`, `correct_answer` and `is_correct`.
- `annotation_stroke`: pen down → pen up. `kept` is False if the stroke was later erased.
- `validation`: the reVISit break slide.
- `validation_markers`: Pupil Capture's marker sequence within the break slide (from `notify.pldata`).
- `calibration` / pre-study `validation`: from Pupil Capture.
- `tlx`: the NASA-TLX slide after a chart (slide shown → submitted). It has the same `block` and `task` as
  its chart, and the six 0–100 ratings as `tlx_mental_demand`, `tlx_physical_demand`, `tlx_temporal_demand`,
  `tlx_performance`, `tlx_effort` and `tlx_frustration`.
  - These are also written to `<pid>_tlx.csv`, one row per chart.
  - reVISit records no per-slider timestamps, only the slide's start/end and raw mouse `windowEvents`.
  - The direction of the performance scale (whether 100 means good or failure) is not in the export, so
    no combined raw-TLX score is computed. This must be checked against the study config.

Each row has start/end on every clock, plus `in_recording`.

### TLX comparison plot (`scripts/tlx_plot.py`)

`uv run python scripts/tlx_plot.py recordings/p2 recordings/p3` writes `recordings/tlx_p2_p3.{png,csv}`.
- Ratings are read straight from each reVISit JSON, so it doesn't need the Pupil recording or a built timeline.
- There is one panel per TLX scale. Rows are chart/task pairs, grouped area → stackarea → line, with one marker
  per participant. Participants are shifted slightly up or down so tied ratings stay visible.
- All study tasks are shown, including stackarea's `-relative` tasks, because TLX was asked after every chart.

## Pupil dilation per task (`scripts/task_pupil_dilation.py`)

Output: `<pid>_pupil_dilation.csv`, one row per task.

Task selection, set in `TASK_ALIASES` and `DROP_TASKS` so that tasks line up across chart types:
- Stackarea's `value-absolute` is analysed as `value`.
- Stackarea's `value-relative` and `comparisons-relative` are not analysed (less of interest).
- The study's own task name is kept in `source_task`.

- **Baseline:** for each eye separately, the mean diameter over the task's 5 s fixation cross
  (cross shown → hidden).
- **Task mean change:** mean over the chart slide minus the baseline, in px and in %. The eye-combined
  % change is the mean of the per-eye % changes.
- **Peak dilation:** the maximum of the eye-averaged % change, computed on 50 ms median bins and
  smoothed with a 250 ms rolling mean. `peak_time_s` is measured from chart onset.

Measure: the **2D** diameter (pupil-ellipse major axis, eye-camera px) is primary. pye3d's 3D diameter
(mm) is kept as `mm3d_*` columns for reference.
- On p1, eye0's 3D model refit repeatedly: its eyeball-centre estimate ranged 29–55 mm from the camera,
  which caused step artifacts.
- The two eyes agree better in 2D: median within-trial correlation r = 0.70 (2D) vs 0.44 (3D), and median
  RMS difference 6.1% vs 12.2% of baseline.
- The 3D loader's 1.5 mm floor would have cut 24% of eye0's valid samples, so this script uses a
  0.5 mm floor.

Interpretation caveats:
- The white chart panel is brighter than the grey fixation screen. Every task starts with a light-reflex
  constriction (−20 to −35% within ~1 s), so task-mean changes mix brightness with cognitive load.
- A peak in the first ~0.5 s is baseline carry-over from before the reflex, not a dilation.

### Data quality (`scripts/pupil_quality.py`)

`<pid>_pupil_quality.csv`: for each task's baseline and chart window, per eye, it records:
- the sampling rate;
- the fraction of samples passing confidence;
- the fraction surviving `clean_eye`, and the fraction surviving Pupil-Labs-only filtering;
- the fraction of 50 ms bins with data;
- the number of Player blinks.

Medians over tasks:

| | eye0 conf ok | eye1 conf ok | eye0 bins (clean_eye) | eye1 bins | blinks |
|---|---|---|---|---|---|
| p1 baseline / chart | 0.98 / 0.96 | 0.98 / 0.99 | 0.90 / 0.83 | 0.89 / 0.88 | 4.2/min |
| p2 baseline / chart | 0.35 / 0.51 | 0.96 / 0.98 | 0.28 / 0.57 | 0.78 / 0.86 | 20.9/min (7% of the time) |

p2's data loss is mostly **eye0 (right eye) tracking**, not blinks:
- The eye0 camera image shows the eye in the bottom-left corner, overexposed, with the pupil near the image edge.
- Median confidence is 0.83 for eye0 vs 1.00 for eye1.
- A 21/min blink rate is within the normal range.

In p2's eye-mean trace, eye1 alone stands in for long stretches. Worst eye0 chart-bin coverage:
area/trends 0.28, area/value 0.29, stackarea/trends 0.33.

## Synced task video (`scripts/sync_video.py`)

`uv run python scripts/sync_video.py recordings/<pid> <block> <task>` writes
`videos/<pid>_<block>-<task>_sync.mp4` at 60 fps on a Pupil-time grid. It shows:
- the world camera with gaze overlaid;
- the tablet screen recording;
- a status strip: time from chart onset, current slide, pen up/down, and right/left/mean pupil % change;
- the dilation trace with a moving cursor.

Alignment:
- The screen recording is delayed by the measured world-camera lag (98 ms on p1), so both video panels
  change slides together. On p1 area/trends this was checked to within 1 frame at chart onset and end.
- The pupil trace and stroke indicator use the timeline's times unchanged.

Convention: eye0 = right eye, eye1 = left eye.

### Filter versions
- Default (`clean_eye`): confidence ≥ 0.8, then speed, trend and island artifact filters.
- `--cleaning pupil_labs` → `*_plfilter.{csv,png}`: confidence ≥ 0.8, then drop samples inside Pupil
  Player's exported blinks (`exports/*/blinks.csv`). No filtering of our own.
- Both versions drop 2D diameters wider than the eye-camera image (192 px). That is physically impossible:
  on p1, one eye0 sample was 1.1e10 px with confidence 0.99.

### Gaze offset correction (`pupil_analysis/gaze.py`)

Gaze is checked against Pupil Capture's 4 validations (marker position in the world image vs gaze), on p1:
- Pupil Player's post-hoc gaze is 29–109 px below the markers, and gets worse and noisier over the session.
- Pupil Capture's recorded gaze is 11–27 px below.

`sync_video.py` therefore draws recorded gaze by default, minus the validation offset interpolated in time.
Player's exported fixations are corrected with the Player-gaze offsets, since they were detected on that gaze.

## Gaze on the screen recording (`scripts/screen_mapping.py`, `pupil_analysis/surface.py`)

`uv run python scripts/screen_mapping.py recordings/<pid>` maps gaze into screen-recording pixels
(content area 2386 x 1668 px, above the letterbox; measured as the first all-black row).

Method:
1. **Tags.** AprilTags 4 (top right), 5 (bottom right), 6 (top left) and 7 (bottom left) sit on the tablet bezel.
   - Pupil Player's detections are filled in with OpenCV's tag36h11 detector (sub-pixel corners), cached in
     `<pid>_tags_opencv.csv`.
   - OpenCV corner k = Player corner `[1, 0, 3, 2][k]`, checked on frames where both detect a tag.
   - On p1, OpenCV adds 12,225 detections. It finds tags in the dark area/line validation frames, where
     Player finds none.
2. **Undistortion.** All world points are undistorted with `world.intrinsics` (radial model, k1 = -0.38).
   OpenCV's default 5 iterations are up to 21 px wrong near the image corners, so the code uses 100.
3. **Tag model.** A fixed set of 16 tag corners in one plane, from the median over frames that see all 4 tags.
4. **Display corners.** During the validation markers the screen is all white.
   - In each such frame with >= 3 tags, the bright quadrilateral's edges are line-fit after undistortion,
     trimming the rounded corners.
   - The corners are mapped into the tag plane and median-combined, then sent to the recording's content corners.
   - The tag model is therefore expressed in screen-recording px.
   - p1: 399/399 frames; each frame lies within 1.3–2.4 px of the median.
5. **Per frame.** Camera pose by PnP on the calibrated camera (>= 2 tags; screen px are a metric plane frame).
   - A free 8-DOF homography (`--method homography`) extrapolates badly when only the two right-hand tags
     are visible: up to 205 px at the far marker, vs 104 px with the pose.
6. **Gaze.** Recorded gaze, offset-corrected at the validations, uses the pose of the world frame it falls in.
   Output: `<pid>_gaze_screen.csv` (`sx`, `sy` in px, top-left origin; `n_tags`, `tags`).
   p1: 97.1% of samples mapped.

Checks (`<pid>_screen_mapping_check.csv/.png`; summary in `<pid>_screen_mapping.json`):
- **Geometry only:** Pupil Capture's world-image detection of each validation marker, mapped to the screen,
  vs the marker centre in the screen recording.
  - 4 tags visible (stackarea validation): 2–10 px.
  - Only tags 4+5 (pre-study, area and line validations; dark exposure): 6–42 px, but 95–104 px at the far
    left marker.
- **Tag-subset simulation:** on 4-tag chart frames, drop tags and compare with the 4-tag pose
  (20 display grid points).
  - 3 tags: median 6–11 px, 95th percentile <= 23 px.
  - 2 tags: median 6–12 px, 95th percentile <= 46 px.
  - During charts, 94% of frames have >= 3 tags and 6% have 2.
- **End to end:** mapped gaze during each fixation cross vs the cross position (1501, 905 px).
  - Median 37 px, max 81 px (15 crosses).
  - Since the geometry is ~5 px there, this is mostly the residual gaze-calibration error.
  - 1 world px is about 4.6 screen px, so this matches the 11–27 world-px validation offsets.

`sync_video.py` draws this gaze on the screen panel: green samples and a red mean ring, for the world
frame shown on the left. It now crops the screen recording to exactly 1668 rows. Before, it used 0.84 of 1964 rows,
which cut the bottom 18 rows of the content.

Not yet done: annotation strokes (canvas coordinates) mapped into screen px, so that gaze and strokes can be compared.

## Provenance records (`pupil_analysis/provenance.py`)

Each output that involves shifts gets a JSON record next to it. Every record includes the time it was made,
the command, and the git commit (with an uncommitted-changes flag).
- `<pid>_sync_params.json` (from `event_timeline.py`):
  - the study clock → Pupil time offset;
  - the world-camera lag and the screen-recording offset, each with its fit quality and detection parameters.
- `videos/<video>.provenance.json` (from `sync_video.py`):
  - every shift applied to that video: the screen offset and extra delay;
  - gaze and fixation corrections, with the offsets per validation and the pixels subtracted during the video;
  - what was deliberately not shifted;
  - the hand-set display parameters;
  - a copy of the sync parameters.

All shifts are measured from the data. Only the parameters behind the measurements are set by hand
(thresholds, crop, padding, eye-label convention).
