import numpy as np

from pupil_analysis.timeline import chart_transitions, crossings, fit_offset, study_events

T0 = 1_000_000_000_000  # Unix ms


def answer(identifier, component, start, end, ans=None, correct=None):
    return {
        "identifier": identifier,
        "componentName": component,
        "startTime": T0 + start,
        "endTime": T0 + end,
        "answer": ans if ans is not None else {},
        "correctAnswer": [{"id": "answer", "answer": correct}] if correct is not None else [],
    }


ANSWERS = [
    answer("introduction_0", "introduction", 0, 1000),
    answer(
        "fixation-area-trends_1",
        "fixation-area-trends",
        1000,
        6100,
        {"baseline": {"crossShownAt": T0 + 1050, "crossHiddenAt": T0 + 6050}},
    ),
    answer(
        "graph-area-trends_2",
        "graph-area-trends",
        6100,
        16100,
        {
            "answer": "Rising",
            "drawing": {
                "elements": [{"id": "a"}, {"id": "b", "isDeleted": True}],
                "markTimes": {"b": {"start": T0 + 9000, "end": T0 + 9500}, "a": {"start": T0 + 7000, "end": T0 + 7400}},
            },
        },
        correct="Rising",
    ),
    answer("validation-break_3", "validation-break", 16100, 30000),
    answer(
        "fixation-line-value_4",
        "fixation-line-value",
        30000,
        35100,
        {"baseline": {"crossShownAt": T0 + 30050, "crossHiddenAt": T0 + 35050}},
    ),
    answer("graph-line-value_5", "graph-line-value", 35100, 40000, {"answer": "$5", "drawing": {}}, correct="$6"),
]


def test_study_events_rows_blocks_and_answers():
    ev = study_events(ANSWERS)
    assert ev.event.tolist() == [
        "fixation_cross", "graph", "annotation_stroke", "annotation_stroke", "validation", "fixation_cross", "graph"
    ]  # fmt: skip
    assert ev.block.tolist() == ["area"] * 5 + ["line"] * 2
    assert ev.block_index.tolist() == [1] * 5 + [2] * 2
    fix = ev.iloc[0]
    assert (fix.start_ms, fix.end_ms) == (T0 + 1050, T0 + 6050)  # cross times, not slide times
    strokes = ev[ev.event == "annotation_stroke"]
    assert strokes.stroke_id.tolist() == ["a", "b"]  # ordered by pen-down time
    assert strokes.kept.tolist() == [True, False]
    graphs = ev[ev.event == "graph"]
    assert graphs.is_correct.tolist() == [True, False]
    assert graphs.task.tolist() == ["trends", "value"]


def test_chart_transitions():
    tr = chart_transitions(study_events(ANSWERS))
    assert tr.t.tolist() == [T0 + 6100, T0 + 16100, T0 + 35100, T0 + 40000]
    assert tr.kind.tolist() == ["chart_on", "chart_off", "chart_on", "chart_off"]


def test_crossings():
    t = np.arange(10.0)
    sig = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 0])
    assert crossings(t, sig, 0.5).tolist() == [2.0, 5.0, 7.0, 8.0]


def test_fit_offset_recovers_shift_and_ignores_spurious_edges():
    rng = np.random.default_rng(0)
    ref = np.sort(rng.uniform(0, 300, 20))
    edges = np.r_[ref + 42.0 + rng.normal(0, 0.01, 20), [5.5, 77.7, 150.1]]  # three extra transitions
    off, res = fit_offset(np.sort(edges), ref, tol=0.2)
    assert abs(off - 42.0) < 0.01
    assert res.edge.notna().all()
    assert res.residual_s.abs().max() < 0.05


def test_fit_offset_marks_unmatched_references():
    ref = np.array([10.0, 20.0, 30.0, 40.0])
    edges = np.array([10.1, 20.1, 30.1])  # last transition never seen
    off, res = fit_offset(edges, ref, tol=0.3, search=1.0)
    assert abs(off - 0.1) < 1e-9
    assert res.edge.isna().tolist() == [False, False, False, True]
