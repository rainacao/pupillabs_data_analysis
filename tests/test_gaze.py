import pandas as pd
import pytest

from pupil_analysis.gaze import correct_gaze


def test_correct_gaze_interpolates_between_validations_and_clamps_outside():
    offsets = pd.DataFrame({"t": [10.0, 20.0], "dx": [0.0, 0.1], "dy": [-0.02, -0.04], "n": [160, 160]})
    gaze = pd.DataFrame({"timestamp": [0.0, 15.0, 30.0], "x": [0.5, 0.5, 0.5], "y": [0.5, 0.5, 0.5]})
    out = correct_gaze(gaze, offsets)
    assert out.x.tolist() == pytest.approx([0.5, 0.45, 0.4])
    assert out.y.tolist() == pytest.approx([0.52, 0.53, 0.54])
    assert gaze.x.tolist() == [0.5, 0.5, 0.5]  # input untouched


def test_correct_gaze_without_validations_is_a_no_op():
    gaze = pd.DataFrame({"timestamp": [1.0], "x": [0.3], "y": [0.7]})
    assert correct_gaze(gaze, pd.DataFrame(columns=["t", "dx", "dy", "n"])).equals(gaze)
