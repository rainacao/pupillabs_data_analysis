"""How small a saccade can pymovements' microsaccade detector find in this recording?

Injects synthetic saccades of known amplitude into the real per-eye signal
(mid-fixation, where the eye is quiet), runs Engbert & Kliegl detection,
and reports the hit rate per amplitude.

    uv run python scripts/microsaccade_detectability.py
"""

import warnings

import numpy as np

from pupil_analysis.io import load_frame_lookup, load_pldata
from pupil_analysis.saccades import detect_saccades, eye_angles

warnings.filterwarnings("ignore")

REC = "recordings/004"
START, END = 4437, 5169  # Player frames: stacked bar chart
AMPLITUDES = (0.25, 0.5, 1.0, 1.5, 2.0)  # degrees
THRESHOLD_FACTORS = (6, 4)  # Engbert & Kliegl's lambda; 6 is pymovements' default
RISE_SAMPLES = 2  # ~16 ms at 124 Hz
HOLD_S = 0.15  # injected step returns after this long so injections don't accumulate




def detect_onsets(grid, x, y, dt, threshold_factor):
    """Onset timestamps (s) of events found by pymovements' microsaccade detector."""
    return detect_saccades(grid, x, y, dt, threshold_factor).onset.to_numpy()


def inject(grid, x, y, dt, sites, amplitude, rng):
    """Copies of x, y with a transient step of `amplitude` deg in a random direction at each site."""
    x, y = x.copy(), y.copy()
    idx = np.arange(len(grid))
    for s in sites:
        k = int(np.searchsorted(grid, s))
        up = np.clip((idx - k) / RISE_SAMPLES, 0, 1)
        down = np.clip((idx - k - int(HOLD_S / dt)) / RISE_SAMPLES, 0, 1)
        angle = rng.uniform(0, 2 * np.pi)
        x += amplitude * np.cos(angle) * (up - down)
        y += amplitude * np.sin(angle) * (up - down)
    return x, y


def main() -> None:
    rng = np.random.default_rng(0)
    rows = load_pldata(f"{REC}/offline_data", "offline_pupil")
    fixations = load_pldata(f"{REC}/offline_data", "fixations")
    frame_ts = load_frame_lookup(REC).timestamp.to_numpy()
    t0, t1 = frame_ts[START], frame_ts[END + 1]
    # every other mid-fixation point, so injections are spaced apart
    sites = [f["timestamp"] + f["duration"] / 2000 for f in fixations if t0 + 1 < f["timestamp"] < t1 - 1][::2]

    for eye in (0, 1):
        grid, x, y, dt = eye_angles(rows, eye, t0, t1)
        print(f"eye {eye}: {1 / dt:.0f} Hz, {len(sites)} injections per amplitude")
        for factor in THRESHOLD_FACTORS:
            print(f"  lambda={factor}: {len(detect_onsets(grid, x, y, dt, factor))} events without injection")
            for amp in AMPLITUDES:
                onsets = detect_onsets(grid, *inject(grid, x, y, dt, sites, amp, rng), dt, factor)
                hits = sum(np.any(np.abs(onsets - s) < 0.03) for s in sites)
                print(f"    {amp:.2f} deg: {hits}/{len(sites)} detected ({100 * hits / len(sites):.0f}%)")


if __name__ == "__main__":
    main()
