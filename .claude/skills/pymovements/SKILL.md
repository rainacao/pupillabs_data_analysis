---
name: pymovements
description: Notes for using pymovements 0.28 (eye-movement event detection) with Pupil Core data in this project — building a Gaze object, velocity, microsaccade/saccade detection, and units gotchas.
---

# pymovements 0.28 with Pupil Core data

Docs: https://pymovements.readthedocs.io/en/stable/ — notes below were checked against the installed 0.28.0 source.

## Building a Gaze object
```python
import pymovements as pm
gaze = pm.gaze.from_pandas(
    df,                                   # columns: time, x, y
    experiment=pm.Experiment(sampling_rate=rate_hz),
    time_column="time", time_unit="ms",   # 's', 'ms' or 'step'; stored internally as ms
    position_columns=["x", "y"],          # degrees of visual angle
)
```
- `pixel_columns=` instead of `position_columns=` needs screen size/distance in `Experiment` and `gaze.pix2deg()`; not meaningful for a head-mounted world camera.
- Samples must be evenly spaced: `pos2vel` uses `sampling_rate`, not the time column. Resample first (`np.interp` onto a uniform grid, NaN over gaps).

## Velocity and detection
```python
gaze.pos2vel(method="smooth")             # also 'fivepoint' (default), 'savitzky_golay', 'neighbors', 'preceding'
gaze.detect("microsaccades", minimum_duration=12, threshold_factor=6)
gaze.compute_event_properties(["amplitude", "peak_velocity"])
events = gaze.events.frame                # polars: name, onset, offset, duration (+ properties)
```
- Detection methods: `microsaccades` (Engbert & Kliegl, noise-adaptive elliptic velocity threshold), `ivt`, `idt`, `ihmm`, `fill`, `blink`, `out_of_screen`.
- **Units gotcha:** `Gaze.detect` passes the time column (ms) as `timesteps`, so `minimum_duration` is in **ms**, not samples. The docstring default of 6 means 6 ms (< 1 sample at 124 Hz).
- `microsaccades` names its events `"saccade"` by default and does not filter by amplitude; it is a general saccade detector. Filter `amplitude <= 1` yourself for microsaccades.
- Threshold internals: `pymovements.events.detection.microsaccades.compute_threshold(velocities, "engbert2015")` returns the per-axis noise SD (deg/s); the detection radius is that times `threshold_factor`.
- `gaze.samples["velocity"]` is a list column that can contain `None`; convert with care before numpy.

## Pupil Core specifics (this project)
- Per-eye rotation: pye3d `pupil.{eye}.3d` records have `circle_3d["normal"]`; yaw = atan2(nx, nz), pitch = atan2(ny, hypot(nx, nz)) in degrees.
- Recording 004 eye cameras ran at 124 Hz; binocular gaze (248 Hz) interleaves the two eyes — detect per eye instead.
- Detectability measured in `scripts/microsaccade_detectability.py`: sub-1° saccades are essentially undetectable in recording 004.
