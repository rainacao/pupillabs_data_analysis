import numpy as np

from pupil_analysis.plots import binned


def test_binned_counts_and_overflow():
    edges, counts = binned(np.array([0, 5, 19.9, 20, 45, 100, 250]), bin_ms=20, max_ms=100)
    assert list(edges) == [0, 20, 40, 60, 80, 100]
    # [0,20): 3, [20,40): 1, [40,60): 1, overflow (>=100): 2
    assert list(counts) == [3, 1, 1, 0, 0, 2]
