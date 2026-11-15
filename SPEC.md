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
