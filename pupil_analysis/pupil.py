"""Pupil diameter, baseline-corrected against the fixation slide before each stimulus.

A trial is [fixation slide][stimulus]. The baseline is the mean diameter over
the last `baseline_s` of the fixation slide; the stimulus response is the
diameter minus that baseline. Each eye is corrected against its own baseline
(pye3d's two eye models disagree on absolute size), then the eyes are averaged.

Frame numbers are Pupil Player's (see `io.load_frame_lookup`).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pupil_analysis.io import load_pldata


@dataclass(frozen=True)
class Trial:
    name: str
    fix_start: int  # first frame of the fixation slide
    onset: int  # first frame of the stimulus (= end of the fixation slide)
    end: int  # first frame after the stimulus


def load_pupil_diameter(
    rec_dir: str | Path,
    min_confidence: float = 0.8,
    valid_mm: tuple[float, float] = (1.5, 9.0),
    measure: str = "3d",
) -> pd.DataFrame:
    """Pupil diameter per sample: columns `timestamp`, `eye`, and `diameter_mm` or `diameter_px`.

    measure="3d": pye3d's `diameter_3d` in mm (`diameter_mm`). Its scale depends on the fitted
    eye model, so it jumps when the model refits. Samples outside `valid_mm` are dropped.
    measure="2d": the pupil ellipse's major axis in eye-camera pixels (`diameter_px`).
    It is model-free, and the major axis is close to the true diameter even when the eye is
    rotated; `valid_mm` is ignored.

    Prefers Pupil Player's post-hoc detection (`offline_data/offline_pupil.pldata`)
    and falls back to Pupil Capture's online `pupil.pldata`. Samples below
    `min_confidence` (blinks, lost tracking) are dropped, then `clean_eye` runs per eye.
    """
    rec_dir = Path(rec_dir)
    offline = rec_dir / "offline_data" / "offline_pupil.pldata"
    rows = load_pldata(offline.parent, offline.stem) if offline.exists() else load_pldata(rec_dir, "pupil")
    rows = [r for r in rows if "diameter_3d" in r]
    if measure == "3d":
        col, values = "diameter_mm", [r["diameter_3d"] for r in rows]
    elif measure == "2d":
        col, values = "diameter_px", [r["diameter"] for r in rows]
    else:
        raise ValueError(f"unknown measure {measure!r}")
    df = pd.DataFrame(
        {
            "timestamp": [r["timestamp"] for r in rows],
            "eye": [r["id"] for r in rows],
            col: values,
            "confidence": [r["confidence"] for r in rows],
        }
    )
    keep = df.confidence >= min_confidence
    if measure == "3d":
        keep &= df[col].between(*valid_mm)
    else:
        keep &= df[col] > 0
    df = df[keep].sort_values("timestamp").reset_index(drop=True)
    cleaned = [clean_eye(g, col=col) for _, g in df.groupby("eye")]
    return pd.concat(cleaned).sort_values("timestamp").reset_index(drop=True)


def _mad_outliers(x: np.ndarray, n_mad: float) -> np.ndarray:
    med = np.nanmedian(x)
    return x > med + n_mad * np.nanmedian(np.abs(x - med))


def _drop_near(eye: pd.DataFrame, bad: np.ndarray, pad_s: float) -> pd.DataFrame:
    """Drop the `bad` samples and every sample within `pad_s` of one."""
    t, bad_t = eye.timestamp.to_numpy(), eye.timestamp.to_numpy()[bad]
    if bad_t.size == 0:
        return eye
    i = np.searchsorted(bad_t, t)
    after = np.abs(bad_t[np.minimum(i, len(bad_t) - 1)] - t)
    before = np.abs(t - bad_t[np.maximum(i - 1, 0)])
    return eye[np.minimum(before, after) > pad_s]


def clean_eye(
    eye: pd.DataFrame,
    speed_n_mad: float = 16.0,
    trend_n_mad: float = 6.0,
    trend_window: str = "500ms",
    gap_s: float = 0.075,
    min_island_s: float = 0.05,
    pad_s: float = 0.05,
    col: str = "diameter_mm",
) -> pd.DataFrame:
    """Remove blink and partial-lid artifacts from one eye's samples (Kret & Sjak-Shie 2019, Behav Res Methods).

    1. dilation speed: larger of backward/forward |Δd/Δt| above median + `speed_n_mad`*MAD
    2. trend deviation: |d - rolling median over `trend_window`| above median + `trend_n_mad`*MAD
    3. islands: runs shorter than `min_island_s` between gaps longer than `gap_s`
    Each step also drops `pad_s` around what it removes.
    """
    eye = eye.sort_values("timestamp")
    if len(eye) < 3:
        return eye
    t, d = eye.timestamp.to_numpy(), eye[col].to_numpy()
    step = np.abs(np.diff(d)) / np.maximum(np.diff(t), 1e-6)
    speed = np.maximum(np.concatenate([[0], step]), np.concatenate([step, [0]]))
    eye = _drop_near(eye, _mad_outliers(speed, speed_n_mad), pad_s)

    s = eye.set_index(pd.to_timedelta(eye.timestamp, unit="s"))[col]
    dev = np.abs((s - s.rolling(trend_window, center=True).median()).to_numpy())
    eye = _drop_near(eye, _mad_outliers(dev, trend_n_mad), pad_s)

    t = eye.timestamp.to_numpy()
    run_id = np.concatenate([[0], np.cumsum(np.diff(t) > gap_s)])
    runs = pd.Series(t).groupby(run_id).agg(["min", "max"])
    short = (runs["max"] - runs["min"]) < min_island_s
    return eye[~short.reindex(run_id).to_numpy()]


def low_segments(signal: np.ndarray, threshold: float, min_len: int) -> list[tuple[int, int]]:
    """[start, end) index ranges where `signal < threshold` for at least `min_len` samples."""
    low = np.concatenate([[False], np.asarray(signal) < threshold, [False]]).astype(int)
    starts, ends = np.flatnonzero(np.diff(low) == 1), np.flatnonzero(np.diff(low) == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_len]


def binned_trace(pupil: pd.DataFrame, t0: float, t1: float, bin_s: float, col: str = "diameter_mm") -> pd.DataFrame:
    """Median diameter per eye in `bin_s` bins over [t0, t1): index = bin start (s from t0), columns = eyes."""
    win = pupil[(pupil.timestamp >= t0) & (pupil.timestamp < t1)]
    n_bins = int(np.ceil((t1 - t0) / bin_s))
    bins = ((win.timestamp - t0) // bin_s).astype(int)
    trace = win.groupby([bins, win.eye])[col].median().unstack("eye")
    return trace.reindex(range(n_bins)).set_axis(np.arange(n_bins) * bin_s)


def trial_traces(
    pupil: pd.DataFrame, trials: list[Trial], frame_ts: np.ndarray, baseline_s: float = 1.0, bin_s: float = 0.05
) -> pd.DataFrame:
    """Long table of baseline-corrected change, one row per (trial, time bin).

    `t` is seconds from stimulus onset (negative = fixation slide).
    `change_eye<i>` is that eye's diameter minus its baseline; `change_mm` is
    the mean of the eyes that have data in the bin.
    """
    out = []
    for tr in trials:
        t_fix, t_on, t_end = frame_ts[tr.fix_start], frame_ts[tr.onset], frame_ts[tr.end]
        base_win = pupil[(pupil.timestamp >= t_on - baseline_s) & (pupil.timestamp < t_on)]
        baseline = base_win.groupby("eye").diameter_mm.mean()
        trace = binned_trace(pupil, t_fix, t_end, bin_s)
        change = trace.sub(baseline, axis=1)
        df = pd.DataFrame({"trial": tr.name, "t": np.round(trace.index + (t_fix - t_on), 6)})
        for eye in (0, 1):
            df[f"change_eye{eye}"] = change[eye].to_numpy() if eye in change else np.nan
        df["change_mm"] = df[["change_eye0", "change_eye1"]].mean(axis=1)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def window_dilation(
    pupil: pd.DataFrame,
    windows: pd.DataFrame,
    col: str = "diameter_mm",
    bin_s: float = 0.05,
    peak_smooth_s: float = 0.25,
    prefix: str = "",
) -> pd.DataFrame:
    """Baseline-corrected pupil change per task, with baseline and task windows given in Pupil time.

    `windows` needs `baseline_start`, `baseline_end`, `task_start`, `task_end` (Pupil s); other
    columns are carried through. `col` is the diameter column (`diameter_mm` or `diameter_px`);
    its suffix names the unit in the output columns, and `prefix` is put in front of every
    output column except the carried-through ones.

    Per eye, against its own baseline: baseline = mean diameter in the baseline window,
    task mean = mean diameter in the task window, change = task mean - baseline, and % change
    = 100 * change / baseline. Eye-combined values average the eyes that have data; the combined
    % change is the mean of the per-eye % changes, so eyes measured on different scales weigh
    equally.

    Peak dilation: the task window is cut into `bin_s` bins (median per eye). Each bin's
    per-eye % change is averaged over the eyes, smoothed with a centered `peak_smooth_s`
    rolling mean, and the maximum is taken (`peak_dilation_pct`). `peak_dilation_<unit>` is
    the eye-averaged change in the same smoothed bin, and `peak_time_s` is seconds from task
    onset to that bin's centre. `task_coverage` is the fraction of bins with data from at
    least one eye.
    """
    unit = col.rsplit("_", 1)[-1]
    smooth = max(1, round(peak_smooth_s / bin_s))
    rows = []
    for w in windows.to_dict("records"):
        base = pupil[(pupil.timestamp >= w["baseline_start"]) & (pupil.timestamp < w["baseline_end"])]
        task = pupil[(pupil.timestamp >= w["task_start"]) & (pupil.timestamp < w["task_end"])]
        b = base.groupby("eye")[col].mean().reindex([0, 1])
        m = task.groupby("eye")[col].mean().reindex([0, 1])
        nb, nt = base.groupby("eye").size(), task.groupby("eye").size()
        change, pct = m - b, 100 * (m - b) / b
        out = {}
        for eye in (0, 1):
            out[f"baseline_eye{eye}_{unit}"] = b[eye]
            out[f"task_mean_eye{eye}_{unit}"] = m[eye]
            out[f"mean_change_eye{eye}_{unit}"] = change[eye]
            out[f"mean_change_eye{eye}_pct"] = pct[eye]
            out[f"n_baseline_eye{eye}"] = int(nb.get(eye, 0))
            out[f"n_task_eye{eye}"] = int(nt.get(eye, 0))
        out[f"mean_change_{unit}"] = change.mean()
        out["mean_change_pct"] = pct.mean()

        trace = binned_trace(pupil, w["task_start"], w["task_end"], bin_s, col).reindex(columns=[0, 1])
        d_unit = trace.sub(b, axis=1).mean(axis=1).rolling(smooth, center=True, min_periods=1).mean()
        d_pct = (
            (100 * trace.sub(b, axis=1).div(b, axis=1)).mean(axis=1).rolling(smooth, center=True, min_periods=1).mean()
        )
        if d_pct.notna().any():
            at = d_pct.idxmax()
            out["peak_dilation_pct"] = d_pct[at]
            out[f"peak_dilation_{unit}"] = d_unit[at]
            out["peak_time_s"] = float(at) + bin_s / 2
        else:
            out["peak_dilation_pct"] = out[f"peak_dilation_{unit}"] = out["peak_time_s"] = np.nan
        out["task_coverage"] = trace.notna().any(axis=1).mean()
        rows.append(dict(w) | {prefix + k: v for k, v in out.items()})
    return pd.DataFrame(rows)


def trial_summary(
    pupil: pd.DataFrame, trials: list[Trial], frame_ts: np.ndarray, baseline_s: float = 1.0, bin_s: float = 0.05
) -> pd.DataFrame:
    """Per trial: baseline per eye, mean and peak-negative change over the stimulus, and data coverage."""
    traces = trial_traces(pupil, trials, frame_ts, baseline_s, bin_s)
    rows = []
    for tr in trials:
        t_on = frame_ts[tr.onset]
        base = pupil[(pupil.timestamp >= t_on - baseline_s) & (pupil.timestamp < t_on)]
        b = base.groupby("eye").diameter_mm.mean()
        stim = traces[(traces.trial == tr.name) & (traces.t >= 0)]
        rows.append(
            {
                "trial": tr.name,
                "fixation_s": round(t_on - frame_ts[tr.fix_start], 2),
                "stimulus_s": round(frame_ts[tr.end] - t_on, 2),
                "baseline_eye0_mm": b.get(0, np.nan),
                "baseline_eye1_mm": b.get(1, np.nan),
                "mean_change_eye0_mm": stim.change_eye0.mean(),
                "mean_change_eye1_mm": stim.change_eye1.mean(),
                "mean_change_mm": stim.change_mm.mean(),
                "min_change_mm": stim.change_mm.min(),
                "max_change_mm": stim.change_mm.max(),
                "pct_change": 100 * stim.change_mm.mean() / b.mean(),
                "coverage": stim.change_mm.notna().mean(),
            }
        )
    return pd.DataFrame(rows)
