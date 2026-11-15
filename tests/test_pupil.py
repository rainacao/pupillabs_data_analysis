import numpy as np
import pandas as pd
import pytest

from pupil_analysis.pupil import Trial, low_segments, trial_summary, trial_traces

FRAME_TS = 100.0 + np.arange(1000) / 10  # 10 fps: frame f -> 100 + f/10 s


def make_pupil():
    # 100 Hz, both eyes. Eye 0: 3.0 mm, eye 1: 4.0 mm; both +0.5 mm from t = 110 s (frame 100) on.
    t = np.arange(100.0, 120.0, 0.01)
    step = np.where(t >= 110.0, 0.5, 0.0)
    return pd.DataFrame(
        {
            "timestamp": np.concatenate([t, t]),
            "eye": np.repeat([0, 1], len(t)),
            "diameter_mm": np.concatenate([3.0 + step, 4.0 + step]),
        }
    ).sort_values("timestamp")


TRIALS = [Trial("a", 50, 100, 150)]  # fixation 105-110 s, stimulus 110-115 s


def test_low_segments():
    sig = np.array([5, 1, 1, 1, 5, 1, 5, 1, 1, 1, 1])
    assert low_segments(sig, threshold=2, min_len=3) == [(1, 4), (7, 11)]


def test_trial_traces_baseline_per_eye():
    tr = trial_traces(make_pupil(), TRIALS, FRAME_TS, baseline_s=1.0, bin_s=0.1)
    assert tr.t.min() == pytest.approx(-5.0)
    fix, stim = tr[tr.t < 0], tr[tr.t >= 0]
    # different absolute sizes per eye, same change after per-eye baseline
    assert np.allclose(fix.change_mm, 0.0)
    assert np.allclose(stim.change_eye0, 0.5) and np.allclose(stim.change_eye1, 0.5)


def test_trial_summary():
    s = trial_summary(make_pupil(), TRIALS, FRAME_TS, baseline_s=1.0, bin_s=0.1).iloc[0]
    assert s.baseline_eye0_mm == pytest.approx(3.0)
    assert s.baseline_eye1_mm == pytest.approx(4.0)
    assert s.mean_change_mm == pytest.approx(0.5)
    assert s.fixation_s == pytest.approx(5.0) and s.stimulus_s == pytest.approx(5.0)
    assert s.coverage == pytest.approx(1.0)


def test_clean_eye_drops_artifacts_and_keeps_signal():
    from pupil_analysis.pupil import clean_eye

    t = np.arange(0, 4, 0.01)
    rng = np.random.default_rng(0)
    d = 3.0 + 0.2 * np.sin(t) + 0.005 * rng.standard_normal(len(t))  # slow genuine change
    d[100] = 1.8  # one-sample partial-lid dip at t = 1.0 s
    d[250:260] -= 0.6  # 100 ms dip at t = 2.5 s
    keep = (t < 3.2) | (t > 3.5) | np.isclose(t, 3.35)  # an isolated sample inside a 300 ms gap
    out = clean_eye(pd.DataFrame({"timestamp": t[keep], "diameter_mm": d[keep]}))
    assert not out.timestamp.between(0.97, 1.03).any()
    assert not out.timestamp.between(2.5, 2.59).any()
    assert not np.isclose(out.timestamp, 3.35).any()
    assert len(out) > 0.8 * keep.sum()


def _synthetic_pupil(col="diameter_px"):
    """Eye0 20 -> 22 (+10%), eye1 10 -> 10.5 (+5%) at t=5, with a +40% bump on both eyes at t=7.0-7.5."""
    t = np.arange(0, 10, 0.005)
    rows = []
    for eye, (base, task) in {0: (20.0, 22.0), 1: (10.0, 10.5)}.items():
        d = np.where(t < 5, base, task)
        d = np.where((t >= 7.0) & (t < 7.5), base * 1.4, d)
        rows.append(pd.DataFrame({"timestamp": t, "eye": eye, col: d}))
    return pd.concat(rows).sort_values("timestamp").reset_index(drop=True)


def test_window_dilation_mean_and_peak():
    from pupil_analysis.pupil import window_dilation

    pupil = _synthetic_pupil()
    win = pd.DataFrame(
        {"name": ["a"], "baseline_start": [0.0], "baseline_end": [5.0], "task_start": [5.0], "task_end": [10.0]}
    )
    r = window_dilation(pupil, win, col="diameter_px").iloc[0]
    assert r["name"] == "a"
    assert r.baseline_eye0_px == pytest.approx(20.0) and r.baseline_eye1_px == pytest.approx(10.0)
    # task mean: 4.5 s at task level + 0.5 s at the bump, per eye
    assert r.mean_change_eye0_px == pytest.approx(0.9 * 2.0 + 0.1 * 8.0)
    assert r.mean_change_eye1_pct == pytest.approx(0.9 * 5.0 + 0.1 * 40.0)
    assert r.mean_change_pct == pytest.approx((r.mean_change_eye0_pct + r.mean_change_eye1_pct) / 2)
    assert r.peak_dilation_pct == pytest.approx(40.0)
    assert r.peak_dilation_px == pytest.approx((8.0 + 4.0) / 2)
    assert 2.0 <= r.peak_time_s <= 2.5
    assert r.task_coverage == pytest.approx(1.0)


def test_window_dilation_prefix_and_missing_eye():
    from pupil_analysis.pupil import window_dilation

    pupil = _synthetic_pupil("diameter_mm")
    pupil = pupil[pupil.eye == 1]
    win = pd.DataFrame({"baseline_start": [0.0], "baseline_end": [5.0], "task_start": [5.0], "task_end": [10.0]})
    r = window_dilation(pupil, win, col="diameter_mm", prefix="mm3d_").iloc[0]
    assert np.isnan(r.mm3d_baseline_eye0_mm) and r.mm3d_n_task_eye0 == 0
    assert r.mm3d_mean_change_pct == pytest.approx(r.mm3d_mean_change_eye1_pct)
