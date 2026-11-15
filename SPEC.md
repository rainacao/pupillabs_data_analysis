# SPEC

Eye-tracking analysis for the VLAT line/area annotation study (reVISit + Pupil Core).

## Data per participant (`recordings/<pid>/`)

- `<NNN>/` — Pupil Core recording (Pupil Capture 3.5.7), with Pupil Player exports in `exports/`.
- `VLAT_*_<participant uuid>.json` — reVISit participant export. Timestamps are unix ms.
- `<pid>_screen.mov` — tablet screen recording (ReplayKit, iPad as a Sidecar display of the Mac).

Study order: introduction, then three chart-type blocks (stackarea, area, line; latin-square order).
Each trial is a 5 s fixation cross followed by a chart with a question and a drawing canvas.
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

p1 results: all 30/30 transitions matched in each video.
- World video lags reVISit by a constant 98 ms (display + camera latency), max residual 48 ms.
- Screen-recording residuals are at most 27 ms.

Event times are **not** shifted by the 98 ms lag. They are the times reVISit issued the stimulus.

## Output: `<pid>_event_timeline.csv`

Built by `uv run python scripts/event_timeline.py recordings/<pid>`. There is one row per event:
- `fixation_cross`: cross shown → hidden.
- `graph`: chart shown → answer submitted. Includes `answer`, `correct_answer` and `is_correct`.
- `annotation_stroke`: pen down → pen up. `kept` is False if the stroke was later erased.
- `validation`: the reVISit break slide.
- `validation_markers`: Pupil Capture's marker sequence within the break slide (from `notify.pldata`).
- `calibration` / pre-study `validation`: from Pupil Capture.

Each row has start/end on every clock, plus `in_recording`.

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
