import numpy as np
import pandas as pd
import pytest

from pupil_analysis.events import count_events, fixations_in_interval, load_fixations_csv, saccades_between

FRAME_TS = 9.0 + 0.01 * np.arange(500)  # frame 100 -> 10.0 s, 130 -> 10.3 s, 160 -> 10.6 s


def make_fixations():
    # three fixations along the x axis at 1 m depth, 0.1 s apart, 200 ms each
    return pd.DataFrame(
        {
            "start_timestamp": [10.0, 10.3, 10.6],
            "duration": [200.0, 200.0, 200.0],
            "start_frame_index": [100, 130, 160],
            "end_frame_index": [120, 150, 180],
            "norm_pos_x": [0.5, 0.6, 0.7],
            "norm_pos_y": [0.5, 0.5, 0.5],
            "gaze_point_3d_x": [0.0, 1000 * np.tan(np.radians(10)), 0.0],
            "gaze_point_3d_y": [0.0, 0.0, 0.0],
            "gaze_point_3d_z": [1000.0, 1000.0, 1000.0],
        }
    )


def test_interval_is_inclusive():
    fx = fixations_in_interval(make_fixations(), 100, 130, FRAME_TS)
    assert list(fx.start_frame_index) == [100, 130]


def test_saccades_between_fixations():
    sc = saccades_between(make_fixations())
    assert len(sc) == 2
    assert sc.gap_ms.to_numpy() == pytest.approx([100.0, 100.0])
    assert sc.amplitude_deg.to_numpy() == pytest.approx([10.0, 10.0])


def test_count_events():
    counts = count_events(make_fixations(), {"a": (100, 150), "b": (300, 400)}, FRAME_TS).set_index("interval")
    assert counts.loc["a", "n_fixations"] == 2
    assert counts.loc["a", "n_saccades"] == 1
    assert counts.loc["b", "n_fixations"] == 0
    assert counts.loc["b", "n_saccades"] == 0


def test_csv_missing_columns(tmp_path):
    path = tmp_path / "fixations.csv"
    pd.DataFrame({"start_timestamp": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_fixations_csv(path)
