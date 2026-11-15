"""Marks drawn on the stimulus (reVISit study JSON) aligned with eye-movement events.

reVISit stores times as Unix epoch milliseconds. Pupil Capture records the
Unix time and the Pupil time of the recording start in `info.player.json`,
which gives the offset between the two clocks (same machine, so the clocks
share a source).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def unix_to_pupil_offset(rec_dir: str | Path) -> float:
    """Seconds to add to a Unix timestamp (s) to get Pupil time."""
    info = json.loads((Path(rec_dir) / "info.player.json").read_text())
    return info["start_time_synced_s"] - info["start_time_system_s"]


def load_study(path: str | Path, offset_s: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trials and marks from a reVISit participant JSON, times in seconds plus `offset_s`.

    trials: `component` (e.g. "graph-treemap"), `chart`, `kind` ("fixation" | "graph"),
    `start`, `end` (for fixation slides, when the cross was shown / hidden).
    marks: `chart`, `mark_id`, `start`, `end`, `duration_ms`, `kept` (False if the
    stroke was later erased), canvas bbox `x`, `y`, `width`, `height`, `n_points`.
    """
    answers = json.loads(Path(path).read_text())["answers"]
    trials, marks = [], []
    for ans in sorted(answers.values(), key=lambda a: a["startTime"]):
        component = ans["componentName"]
        kind, chart = component.split("-", 1)
        start, end = ans["startTime"], ans["endTime"]
        answer = ans["answer"] if isinstance(ans["answer"], dict) else {}
        if kind == "fixation" and "baseline" in answer:
            start, end = answer["baseline"]["crossShownAt"], answer["baseline"]["crossHiddenAt"]
        trials.append({"component": component, "chart": chart, "kind": kind, "start": start, "end": end})
        drawing = answer.get("drawing") or {}
        elements = {e["id"]: e for e in drawing.get("elements", []) if not e.get("isDeleted")}
        for mark_id, mt in (drawing.get("markTimes") or {}).items():
            e = elements.get(mark_id, {})
            marks.append(
                {
                    "chart": chart,
                    "mark_id": mark_id,
                    "start": mt["start"],
                    "end": mt["end"],
                    "kept": mark_id in elements,
                    "x": e.get("x", np.nan),
                    "y": e.get("y", np.nan),
                    "width": e.get("width", np.nan),
                    "height": e.get("height", np.nan),
                    "n_points": len(e.get("points", [])) if e else np.nan,
                }
            )
    trials_df = pd.DataFrame(trials)
    marks_df = pd.DataFrame(marks).sort_values("start").reset_index(drop=True)
    for df in (trials_df, marks_df):
        df[["start", "end"]] = df[["start", "end"]] / 1000 + offset_s
    marks_df["duration_ms"] = (marks_df.end - marks_df.start) * 1000
    return trials_df, marks_df


def _overlap(a0: np.ndarray, a1: np.ndarray, b0: float, b1: float) -> np.ndarray:
    return np.clip(np.minimum(a1, b1) - np.maximum(a0, b0), 0, None)


def state_at(t: float, fixations: pd.DataFrame, saccades: pd.DataFrame) -> str:
    """'saccade', 'fixation' or 'other' (blink, smooth pursuit, unclassified) at Pupil time `t`."""
    if ((saccades.onset <= t) & (saccades.offset >= t)).any():
        return "saccade"
    if ((fixations.start <= t) & (fixations.end >= t)).any():
        return "fixation"
    return "other"


def mark_eye_events(marks: pd.DataFrame, fixations: pd.DataFrame, saccades: pd.DataFrame) -> pd.DataFrame:
    """Per mark: eye state at stroke onset/offset, fixation/saccade activity during the stroke, nearest saccades.

    `fixations` need `start`, `end` (Pupil s); `saccades` need `onset`, `offset`.
    """
    sac_on = saccades.onset.to_numpy()
    rows = []
    for m in marks.to_dict("records"):
        dur = m["end"] - m["start"]
        fix_ov = _overlap(fixations.start.to_numpy(), fixations.end.to_numpy(), m["start"], m["end"])
        sac_ov = _overlap(saccades.onset.to_numpy(), saccades.offset.to_numpy(), m["start"], m["end"])
        before = sac_on[sac_on < m["start"]]
        after = sac_on[sac_on > m["end"]]
        rows.append(
            {
                "mark_id": m["mark_id"],
                "state_at_start": state_at(m["start"], fixations, saccades),
                "state_at_end": state_at(m["end"], fixations, saccades),
                "n_fixations_during": int((fix_ov > 0).sum()),
                "fixation_coverage": fix_ov.sum() / dur if dur > 0 else np.nan,
                "n_saccades_during": int((sac_ov > 0).sum()),
                "saccade_rate_during_hz": (sac_ov > 0).sum() / dur if dur > 0 else np.nan,
                "last_saccade_before_ms": (m["start"] - before[-1]) * 1000 if before.size else np.nan,
                "first_saccade_after_ms": (after[0] - m["end"]) * 1000 if after.size else np.nan,
            }
        )
    return marks.merge(pd.DataFrame(rows), on="mark_id")


def peri_event_rates(
    events: np.ndarray, align: np.ndarray, window: tuple[float, float] = (-2.0, 2.0), bin_s: float = 0.1
) -> pd.DataFrame:
    """Rate (Hz) of `events` (onset times) in bins around each `align` time, averaged over alignments.

    Columns: `t` (bin start relative to the alignment), `rate_hz`, `sem_hz`.
    """
    edges = np.arange(window[0], window[1] + bin_s / 2, bin_s)
    counts = np.array([np.histogram(events - a, edges)[0] for a in align]) / bin_s
    return pd.DataFrame(
        {
            "t": edges[:-1],
            "rate_hz": counts.mean(0),
            "sem_hz": counts.std(0, ddof=1) / np.sqrt(len(align)) if len(align) > 1 else np.nan,
        }
    )


def drawing_vs_not(
    trials: pd.DataFrame, marks: pd.DataFrame, fixations: pd.DataFrame, saccades: pd.DataFrame
) -> pd.DataFrame:
    """Per chart: saccade rate and fixation coverage while drawing vs. the rest of the chart time."""
    rows = []
    fs, fe, so = fixations.start.to_numpy(), fixations.end.to_numpy(), saccades.onset.to_numpy()
    for tr in trials[trials.kind == "graph"].itertuples():
        grid = np.arange(tr.start, tr.end, 0.005)  # 5 ms grid for time-weighted measures
        drawing = np.zeros(len(grid), bool)
        for m in marks[marks.chart == tr.chart].itertuples():
            drawing |= (grid >= m.start) & (grid < m.end)
        i = np.searchsorted(fs, grid, side="right") - 1
        in_fix = (i >= 0) & (grid < fe[np.maximum(i, 0)])
        j = np.clip(np.searchsorted(grid, so[(so >= tr.start) & (so < tr.end)]), 0, len(grid) - 1)
        for label, mask in (("drawing", drawing), ("not drawing", ~drawing)):
            secs = mask.sum() * 0.005
            rows.append(
                {
                    "chart": tr.chart,
                    "phase": label,
                    "seconds": secs,
                    "n_saccades": int(mask[j].sum()),
                    "saccade_rate_hz": mask[j].sum() / secs if secs > 0 else np.nan,
                    "fixation_coverage": in_fix[mask].mean() if mask.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)
