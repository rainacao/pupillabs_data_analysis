"""Fixation and saccade counts per frame interval.

Pupil Core detects fixations but not saccades. Here a saccade is the
transition between two consecutive fixations inside the interval, so
n_saccades = n_fixations - 1 (when there is at least one fixation).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pupil_analysis.io import load_frame_lookup, load_pldata, timestamps_to_frames

FIXATION_COLUMNS = ["start_timestamp", "duration", "start_frame_index", "end_frame_index", "norm_pos_x", "norm_pos_y"]


def load_fixations_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load `fixations.csv` from a Pupil Player export (duration in ms)."""
    df = pd.read_csv(csv_path)
    missing = set(FIXATION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns {sorted(missing)}; is it a Pupil Player fixations export?")
    return df.sort_values("start_timestamp").reset_index(drop=True)


def load_fixations_pldata(rec_dir: str | Path, offline: bool = True) -> pd.DataFrame:
    """Fixations from .pldata, in the same columns as the CSV export.

    offline=True reads Pupil Player's offline detector results
    (`offline_data/fixations.pldata`); offline=False reads Pupil Capture's
    online detector, which re-publishes each fixation as it grows (the last
    update per id is kept) and reports coarse durations.
    """
    rows = load_pldata(Path(rec_dir) / "offline_data" if offline else rec_dir, "fixations")
    df = pd.DataFrame(
        {
            "id": [r["id"] for r in rows],
            "start_timestamp": [r["timestamp"] for r in rows],
            "duration": [r["duration"] for r in rows],
            "norm_pos_x": [r["norm_pos"][0] for r in rows],
            "norm_pos_y": [r["norm_pos"][1] for r in rows],
            "gaze_point_3d_x": [r["gaze_point_3d"][0] for r in rows],
            "gaze_point_3d_y": [r["gaze_point_3d"][1] for r in rows],
            "gaze_point_3d_z": [r["gaze_point_3d"][2] for r in rows],
        }
    )
    df = df.sort_values("duration").groupby("id").tail(1).sort_values("start_timestamp")
    frame_ts = load_frame_lookup(rec_dir).timestamp.to_numpy()
    df["start_frame_index"] = timestamps_to_frames(df.start_timestamp, frame_ts)
    df["end_frame_index"] = timestamps_to_frames(df.start_timestamp + df.duration / 1000, frame_ts)
    return df.reset_index(drop=True)


def fixations_in_interval(fixations: pd.DataFrame, start: int, end: int, frame_ts: np.ndarray) -> pd.DataFrame:
    """Fixations starting within Player frames `start`..`end` (inclusive).

    Selection is by timestamp (`frame_ts` = `load_frame_lookup(...).timestamp`),
    so it doesn't depend on which frame numbering a CSV used.
    """
    t0 = frame_ts[start]
    t1 = frame_ts[end + 1] if end + 1 < len(frame_ts) else np.inf
    mask = (fixations.start_timestamp >= t0) & (fixations.start_timestamp < t1)
    return fixations[mask].sort_values("start_timestamp").reset_index(drop=True)


def saccades_between(fixations: pd.DataFrame) -> pd.DataFrame:
    """One row per transition between consecutive fixations.

    Amplitude is the angle between the two fixations' 3D gaze points when
    those columns exist, otherwise the distance in normalized image units.
    """
    a, b = fixations.iloc[:-1].reset_index(drop=True), fixations.iloc[1:].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "start_timestamp": a.start_timestamp + a.duration / 1000,
            "end_timestamp": b.start_timestamp,
        }
    )
    out["gap_ms"] = (out.end_timestamp - out.start_timestamp) * 1000
    cols = ["gaze_point_3d_x", "gaze_point_3d_y", "gaze_point_3d_z"]
    if set(cols) <= set(fixations.columns):
        va, vb = a[cols].to_numpy(float), b[cols].to_numpy(float)
        cos = (va * vb).sum(1) / (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1))
        out["amplitude_deg"] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    else:
        out["amplitude_norm"] = np.hypot(b.norm_pos_x - a.norm_pos_x, b.norm_pos_y - a.norm_pos_y)
    return out


def count_events(fixations: pd.DataFrame, intervals: dict[str, tuple[int, int]], frame_ts: np.ndarray) -> pd.DataFrame:
    """Fixation and saccade counts (plus basic stats) for each named Player frame interval."""
    rows = []
    for name, (start, end) in intervals.items():
        fx = fixations_in_interval(fixations, start, end, frame_ts)
        sc = saccades_between(fx)
        amp = sc.get("amplitude_deg")
        rows.append(
            {
                "interval": name,
                "start_frame": start,
                "end_frame": end,
                "n_fixations": len(fx),
                "n_saccades": len(sc),
                "mean_fixation_ms": fx.duration.mean(),
                "total_fixation_ms": fx.duration.sum(),
                "mean_saccade_amplitude_deg": amp.mean() if amp is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)
