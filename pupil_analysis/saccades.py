"""Velocity-based saccade detection on eye-in-head rotation (pye3d pupil normals).

Each eye is detected separately with Engbert & Kliegl's adaptive velocity
threshold (pymovements' `microsaccades` detector, which finds saccades of any
size). `binocular_saccades` keeps events that overlap in time in both eyes,
Engbert & Kliegl's (2003) criterion against monocular noise.
"""

import warnings

import numpy as np
import pandas as pd
import pymovements as pm


def eye_angles(rows: list[dict], eye: int, t0: float, t1: float, min_confidence: float = 0.6):
    """Uniformly resampled eye-rotation angles (deg) from pye3d pupil normals; gaps left as NaN.

    `rows` are pupil records (e.g. `load_pldata(rec/"offline_data", "offline_pupil")`).
    Returns (grid timestamps, yaw, pitch, dt).
    """
    r = [x for x in rows if x["topic"] == f"pupil.{eye}.3d" and x["confidence"] >= min_confidence]
    t = np.array([x["timestamp"] for x in r])
    n = np.array([x["circle_3d"]["normal"] for x in r]).reshape(-1, 3)
    order = np.argsort(t)
    t, n = t[order], n[order]
    keep = (t >= t0) & (t < t1)
    t, n = t[keep], n[keep]
    yaw = np.degrees(np.arctan2(n[:, 0], n[:, 2]))
    pitch = np.degrees(np.arctan2(n[:, 1], np.hypot(n[:, 0], n[:, 2])))
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1], dt)
    x, y = np.interp(grid, t, yaw), np.interp(grid, t, pitch)
    observed = np.abs(t[np.clip(np.searchsorted(t, grid), 0, len(t) - 1)] - grid) < dt
    x[~observed] = np.nan
    y[~observed] = np.nan
    return grid, x, y, dt


def detect_saccades(
    grid: np.ndarray, x: np.ndarray, y: np.ndarray, dt: float, threshold_factor: float = 6, minimum_duration_ms=12
) -> pd.DataFrame:
    """Saccades in one eye's signal: `onset`, `offset` (Pupil s), `amplitude_deg`, `peak_velocity`."""
    df = pd.DataFrame({"time": (grid - grid[0]) * 1000, "x": x, "y": y})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gaze = pm.gaze.from_pandas(
            df,
            experiment=pm.Experiment(sampling_rate=1 / dt),
            time_column="time",
            time_unit="ms",
            position_columns=["x", "y"],
        )
        gaze.pos2vel(method="smooth")
        # detect() passes time in ms, so minimum_duration is in ms (12 ms as in Engbert & Kliegl 2003)
        gaze.detect("microsaccades", minimum_duration=minimum_duration_ms, threshold_factor=threshold_factor)
        gaze.compute_event_properties(["amplitude", "peak_velocity"])
    ev = gaze.events.frame.to_pandas()
    return pd.DataFrame(
        {
            "onset": ev.onset.to_numpy() / 1000 + grid[0],
            "offset": ev.offset.to_numpy() / 1000 + grid[0],
            "amplitude_deg": ev.amplitude.to_numpy(),
            "peak_velocity": ev.peak_velocity.to_numpy(),
        }
    )


def binocular_saccades(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Events present in both eyes: each `left` event that overlaps a `right` event, merged.

    onset/offset span both eyes' events; amplitude and peak velocity are the
    mean of the two eyes.
    """
    rows = []
    for a in left.to_dict("records"):
        hit = right[(right.onset <= a["offset"]) & (right.offset >= a["onset"])]
        if hit.empty:
            continue
        b = hit.iloc[0]
        rows.append(
            {
                "onset": min(a["onset"], b.onset),
                "offset": max(a["offset"], b.offset),
                "amplitude_deg": (a["amplitude_deg"] + b.amplitude_deg) / 2,
                "peak_velocity": (a["peak_velocity"] + b.peak_velocity) / 2,
            }
        )
    out = pd.DataFrame(rows, columns=["onset", "offset", "amplitude_deg", "peak_velocity"])
    return out.drop_duplicates(["onset", "offset"]).reset_index(drop=True)
