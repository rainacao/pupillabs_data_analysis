"""Align the marks drawn in the reVISit study (JSON, Unix ms) with fixations and saccades (Pupil time).

uv run python scripts/marks_vs_eye_events.py

Writes to recordings/003/marks/:
  trials.csv              chart / fixation-slide intervals from the JSON, in Pupil time (+ Player frames)
  marks.csv               one row per mark: timing, Player frames, eye state at start/end, events during it
  event_timeline.csv      fixations, saccades and marks on one Pupil-time timeline, tagged by chart
  drawing_vs_not.csv      saccade rate and fixation coverage while drawing vs. not, per chart
  mark_timelines.png      per chart: fixations, saccades and marks over time
  peri_mark_rates.png     saccade/fixation onset rates around mark start and end
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pupil_analysis.events import load_fixations_pldata
from pupil_analysis.io import load_frame_lookup, load_gaze, load_pldata, timestamps_to_frames
from pupil_analysis.marks import drawing_vs_not, load_study, mark_eye_events, peri_event_rates, unix_to_pupil_offset
from pupil_analysis.plots import plot_mark_timelines, plot_peri_mark
from pupil_analysis.saccades import binocular_saccades, detect_saccades, eye_angles

REC = Path("recordings/003")
STUDY_JSON = next(Path("recordings/annotations").glob("003_*.json"))
WINDOW, BIN_S = (-2.0, 2.0), 0.2
GAZE_CONFIDENCE = 0.85  # Player's fixation detector threshold (offline_data/fixations.meta)


def chart_of(t: np.ndarray, trials: pd.DataFrame) -> list:
    """Component name ("graph-treemap", "fixation-pie", ...) each time falls in, else None."""
    out = []
    for x in t:
        hit = trials[(trials.start <= x) & (trials.end > x)]
        out.append(hit.component.iloc[0] if len(hit) else None)
    return out


def main() -> None:
    out_dir = REC / "marks"
    out_dir.mkdir(parents=True, exist_ok=True)
    offset = unix_to_pupil_offset(REC)
    trials, marks = load_study(STUDY_JSON, offset)
    frame_ts = load_frame_lookup(REC).timestamp.to_numpy()
    t0, t1 = trials.start.min() - 1, trials.end.max() + 1
    print(f"Unix -> Pupil offset {offset:.6f} s; study spans Pupil {t0 + 1:.2f}-{t1 - 1:.2f} s")

    fx = load_fixations_pldata(REC)
    fx = fx.assign(start=fx.start_timestamp, end=fx.start_timestamp + fx.duration / 1000)
    fx = fx[(fx.end > t0) & (fx.start < t1)].reset_index(drop=True)

    rows = load_pldata(REC / "offline_data", "offline_pupil")
    per_eye = [detect_saccades(*eye_angles(rows, eye, t0, t1)) for eye in (0, 1)]
    sac = binocular_saccades(*per_eye)
    print(
        f"{len(fx)} fixations, {len(sac)} binocular saccades (eye0 {len(per_eye[0])}, eye1 {len(per_eye[1])}), "
        f"{len(marks)} marks ({(~marks.kept).sum()} later erased)"
    )

    for df, cols in ((trials, ("start", "end")), (marks, ("start", "end"))):
        for c in cols:
            df[f"{c}_frame"] = timestamps_to_frames(df[c], frame_ts)
    trials.to_csv(out_dir / "trials.csv", index=False)

    per_mark = mark_eye_events(marks, fx, sac)
    # fixations need gaze confidence >= 0.85; tracking degrades while looking down at the pen
    gaze = load_gaze(REC, min_confidence=0.0)
    per_mark["gaze_valid_frac"] = [
        (gaze.confidence[(gaze.timestamp >= a) & (gaze.timestamp < b)] >= GAZE_CONFIDENCE).mean()
        for a, b in zip(per_mark.start, per_mark.end)
    ]
    per_mark.to_csv(out_dir / "marks.csv", index=False)

    timeline = (
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "type": "fixation",
                        "start": fx.start,
                        "end": fx.end,
                        "duration_ms": fx.duration,
                        "norm_pos_x": fx.norm_pos_x,
                        "norm_pos_y": fx.norm_pos_y,
                    }
                ),
                pd.DataFrame(
                    {
                        "type": "saccade",
                        "start": sac.onset,
                        "end": sac.offset,
                        "duration_ms": (sac.offset - sac.onset) * 1000,
                        "amplitude_deg": sac.amplitude_deg,
                        "peak_velocity": sac.peak_velocity,
                    }
                ),
                pd.DataFrame(
                    {
                        "type": "mark",
                        "start": marks.start,
                        "end": marks.end,
                        "duration_ms": marks.duration_ms,
                        "mark_id": marks.mark_id,
                        "kept": marks.kept,
                    }
                ),
            ]
        )
        .sort_values("start")
        .reset_index(drop=True)
    )
    timeline.insert(1, "component", chart_of(timeline.start.to_numpy(), trials))
    timeline.insert(2, "start_frame", timestamps_to_frames(timeline.start, frame_ts))
    timeline.to_csv(out_dir / "event_timeline.csv", index=False)

    dvn = drawing_vs_not(trials, marks, fx, sac)
    dvn.to_csv(out_dir / "drawing_vs_not.csv", index=False)

    graphs = trials[trials.kind == "graph"]
    chart_secs = (graphs.end - graphs.start).sum()
    in_chart = lambda t: np.array([((graphs.start <= x) & (graphs.end > x)).any() for x in t])
    baselines = {"saccade": in_chart(sac.onset).sum() / chart_secs, "fixation": in_chart(fx.start).sum() / chart_secs}
    curves = {
        align: {
            "saccade": peri_event_rates(sac.onset.to_numpy(), marks[align].to_numpy(), WINDOW, BIN_S),
            "fixation": peri_event_rates(fx.start.to_numpy(), marks[align].to_numpy(), WINDOW, BIN_S),
        }
        for align in ("start", "end")
    }
    plot_mark_timelines(trials, marks, fx, sac).savefig(out_dir / "mark_timelines.png", dpi=150)
    plot_peri_mark(curves, baselines, len(marks)).savefig(out_dir / "peri_mark_rates.png", dpi=150)

    pooled = dvn.groupby("phase").agg(seconds=("seconds", "sum"), n_saccades=("n_saccades", "sum"))
    pooled["saccade_rate_hz"] = pooled.n_saccades / pooled.seconds
    pooled["fixation_coverage"] = dvn.assign(w=dvn.fixation_coverage * dvn.seconds).groupby("phase").w.sum() / (
        pooled.seconds
    )
    print("\nAll charts pooled:\n" + pooled.round(3).to_string())
    print("\nEye state at mark start:", per_mark.state_at_start.value_counts().to_dict())
    print("Eye state at mark end:  ", per_mark.state_at_end.value_counts().to_dict())
    print(f"Marks with >=1 saccade during the stroke: {(per_mark.n_saccades_during > 0).sum()}/{len(per_mark)}")
    print(
        "Median gaze_valid_frac during marks by state at start:",
        per_mark.groupby("state_at_start").gaze_valid_frac.median().round(2).to_dict(),
    )
    print("Last saccade before mark start (ms), median:", round(per_mark.last_saccade_before_ms.median()))
    print(f"wrote {out_dir}/")


if __name__ == "__main__":
    main()
