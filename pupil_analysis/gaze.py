"""Gaze accuracy offsets measured at Pupil Capture validations, and their correction.

During a validation, Pupil Capture shows markers and logs where each one was detected in the
world image (`validation.data` notifications, `ref_list`). The gaze at those moments should land
on the marker, so `gaze - marker` is the mapping's offset at that point in the session.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pupil_analysis.io import load_pldata


def validation_offsets(rec_dir: str | Path, gaze: pd.DataFrame, max_dt: float = 0.02) -> pd.DataFrame:
    """Median gaze - marker offset per validation, in normalized world coordinates.

    `gaze` needs `timestamp`, `x`, `y` (norm_pos). Each marker sample is paired with the nearest
    gaze sample within `max_dt` s. Columns: `t` (validation mid-time, Pupil s), `dx`, `dy`, `n`.
    """
    gt, gx, gy = gaze.timestamp.to_numpy(), gaze.x.to_numpy(), gaze.y.to_numpy()
    rows = []
    for note in load_pldata(rec_dir, "notify"):
        if note.get("subject") != "validation.data":
            continue
        ref = note["ref_list"]
        rt = np.array([r["timestamp"] for r in ref])
        rx = np.array([r["norm_pos"][0] for r in ref])
        ry = np.array([r["norm_pos"][1] for r in ref])
        i = np.clip(np.searchsorted(gt, rt), 1, len(gt) - 1)
        j = np.where(np.abs(gt[i - 1] - rt) < np.abs(gt[i] - rt), i - 1, i)
        ok = np.abs(gt[j] - rt) <= max_dt
        if ok.sum() == 0:
            continue
        rows.append(
            {
                "t": float(np.median(rt)),
                "dx": float(np.median(gx[j][ok] - rx[ok])),
                "dy": float(np.median(gy[j][ok] - ry[ok])),
                "n": int(ok.sum()),
            }
        )
    return pd.DataFrame(rows, columns=["t", "dx", "dy", "n"]).sort_values("t").reset_index(drop=True)


def correct_gaze(df: pd.DataFrame, offsets: pd.DataFrame, t_col: str = "timestamp") -> pd.DataFrame:
    """Subtract the validation offset from `x`, `y`, interpolated linearly in time between validations.

    Before the first / after the last validation, that validation's offset is used.
    """
    if offsets.empty:
        return df
    t = df[t_col].to_numpy()
    out = df.copy()
    out["x"] = df.x - np.interp(t, offsets.t, offsets.dx)
    out["y"] = df.y - np.interp(t, offsets.t, offsets.dy)
    return out
