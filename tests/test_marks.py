import json

import numpy as np
import pandas as pd
import pytest

from pupil_analysis.marks import drawing_vs_not, load_study, mark_eye_events, peri_event_rates, state_at
from pupil_analysis.saccades import binocular_saccades

T0 = 1_000_000_000_000  # Unix ms


def write_study(tmp_path):
    answers = {
        "fixation-bar_0": {
            "componentName": "fixation-bar",
            "startTime": T0,
            "endTime": T0 + 5100,
            "answer": {"baseline": {"crossShownAt": T0 + 50, "crossHiddenAt": T0 + 5050}},
        },
        "graph-bar_1": {
            "componentName": "graph-bar",
            "startTime": T0 + 5100,
            "endTime": T0 + 15100,
            "answer": {
                "drawing": {
                    "elements": [
                        {"id": "a", "x": 1, "y": 2, "width": 3, "height": 4, "points": [[0, 0], [1, 1]]},
                        {"id": "gone", "isDeleted": True},
                    ],
                    "markTimes": {
                        "a": {"start": T0 + 7000, "end": T0 + 7500},
                        "gone": {"start": T0 + 9000, "end": T0 + 9200},
                    },
                }
            },
        },
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"answers": answers}))
    return path


def test_load_study_converts_unix_ms(tmp_path):
    offset = -T0 / 1000 + 100.0  # Unix T0 -> Pupil 100 s
    trials, marks = load_study(write_study(tmp_path), offset)
    assert list(trials.kind) == ["fixation", "graph"]
    assert trials.start.tolist() == pytest.approx([100.05, 105.1])  # fixation uses crossShownAt
    assert marks.start.tolist() == pytest.approx([107.0, 109.0])
    assert marks.duration_ms.tolist() == pytest.approx([500, 200])
    assert marks.kept.tolist() == [True, False]
    assert marks.n_points.iloc[0] == 2


FIX = pd.DataFrame({"start": [106.8, 107.3], "end": [107.1, 107.6]})
SAC = pd.DataFrame({"onset": [106.5, 107.15, 108.0], "offset": [106.54, 107.2, 108.03]})


def test_state_and_mark_events(tmp_path):
    _, marks = load_study(write_study(tmp_path), -T0 / 1000 + 100.0)
    assert state_at(107.0, FIX, SAC) == "fixation"
    assert state_at(107.17, FIX, SAC) == "saccade"
    assert state_at(107.25, FIX, SAC) == "other"
    m = mark_eye_events(marks, FIX, SAC).set_index("mark_id").loc["a"]
    assert m.state_at_start == "fixation" and m.state_at_end == "fixation"
    assert m.n_fixations_during == 2
    assert m.fixation_coverage == pytest.approx((0.1 + 0.2) / 0.5)
    assert m.n_saccades_during == 1
    assert m.last_saccade_before_ms == pytest.approx(500)
    assert m.first_saccade_after_ms == pytest.approx(500)


def test_drawing_vs_not(tmp_path):
    trials, marks = load_study(write_study(tmp_path), -T0 / 1000 + 100.0)
    d = drawing_vs_not(trials, marks, FIX, SAC).set_index("phase")
    assert d.loc["drawing", "seconds"] == pytest.approx(0.7, abs=0.01)
    assert d.loc["drawing", "n_saccades"] == 1
    assert d.loc["not drawing", "n_saccades"] == 2  # 106.5 and 108.0: in the chart, outside both strokes


def test_peri_event_rates():
    r = peri_event_rates(np.array([0.95, 2.05]), np.array([1.0, 2.0]), window=(-0.2, 0.2), bin_s=0.1)
    assert r.t.tolist() == pytest.approx([-0.2, -0.1, 0.0, 0.1])
    assert r.rate_hz.tolist() == pytest.approx([0, 5, 5, 0])


def test_binocular_requires_overlap():
    left = pd.DataFrame({"onset": [1.0, 2.0], "offset": [1.05, 2.05], "amplitude_deg": [4, 4], "peak_velocity": [1, 1]})
    right = pd.DataFrame({"onset": [1.02], "offset": [1.06], "amplitude_deg": [6], "peak_velocity": [3]})
    b = binocular_saccades(left, right)
    assert len(b) == 1
    assert b.iloc[0].tolist() == pytest.approx([1.0, 1.06, 5, 2])
