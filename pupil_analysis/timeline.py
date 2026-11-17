"""Per-participant event timeline: reVISit study events on the Pupil, world-video and screen-recording clocks.

Clocks
------
- unix ms: reVISit's timestamps (`Date.now()` in the browser).
- Pupil time (s): Pupil Capture's synced clock; eye, gaze and fixation data use it.
  The browser ran on the same Mac as Pupil Capture (the tablet is a Sidecar display),
  so `pupil = unix / 1000 + (start_time_synced_s - start_time_system_s)` from
  `info.player.json`.
- world time (s): seconds since the first world-video frame, i.e. Pupil Player's
  timeline (`pupil - world_timestamps[0]`). `world_frame` is Player's frame index.
- screen time (s): position in the tablet's screen recording. The screen recording has no
  usable clock of its own, so its offset is measured by matching slide transitions
  (grey fixation slide <-> white chart panel) to the reVISit transition times.

The same transition matching against the world video checks the unix -> Pupil
conversion (see `fit_offset`).
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from pupil_analysis.io import load_frame_lookup, load_pldata, timestamps_to_frames
from pupil_analysis.marks import unix_to_pupil_offset

TLX_SCALES = ["mental-demand", "physical-demand", "temporal-demand", "performance", "effort", "frustration"]
TLX_COLUMNS = [f"tlx_{s.replace('-', '_')}" for s in TLX_SCALES]


def load_answers(study_json: str | Path) -> list[dict]:
    """reVISit answers sorted by start time."""
    return sorted(json.loads(Path(study_json).read_text())["answers"].values(), key=lambda a: a["startTime"])


def study_events(answers: list[dict]) -> pd.DataFrame:
    """One row per study event, times in unix ms.

    Events: `fixation_cross` (cross shown -> hidden), `graph` (chart slide shown -> answer
    submitted), `annotation_stroke` (pen down -> pen up; `kept` False if later erased),
    `validation` (the validation-break slide, start -> end), `tlx` (the NASA-TLX slide after a chart,
    start -> submit; the six 0-100 ratings as `tlx_<scale>` columns; only in studies that ask it).
    `block` is the chart type of the block the event belongs to (stackarea/area/line); a validation
    break belongs to the block it follows. `block_index` counts blocks in presentation order.
    """
    rows = []
    block, block_index = None, 0
    for a in answers:
        name = a["componentName"]
        kind, _, rest = name.partition("-")
        if kind in ("fixation", "graph", "tlx"):
            chart_type, _, task = rest.partition("-")
            if chart_type != block:
                block, block_index = chart_type, block_index + 1
        else:
            task = ""
        base = {"block_index": block_index, "block": block, "task": task, "component": a["identifier"]}
        answer = a["answer"] if isinstance(a["answer"], dict) else {}
        if kind == "fixation":
            bl = answer["baseline"]
            rows.append(
                base | {"event": "fixation_cross", "start_ms": bl["crossShownAt"], "end_ms": bl["crossHiddenAt"]}
            )
        elif kind == "graph":
            given = answer.get("answer")
            correct = next((c["answer"] for c in a.get("correctAnswer", []) if c["id"] == "answer"), None)
            rows.append(
                base
                | {
                    "event": "graph",
                    "start_ms": a["startTime"],
                    "end_ms": a["endTime"],
                    "answer": given,
                    "correct_answer": correct,
                    "is_correct": None if correct is None else str(given) == str(correct),
                }
            )
            drawing = answer.get("drawing") or {}
            live = {e["id"] for e in drawing.get("elements", []) if not e.get("isDeleted")}
            strokes = sorted((drawing.get("markTimes") or {}).items(), key=lambda kv: kv[1]["start"])
            for n, (mark_id, mt) in enumerate(strokes, 1):
                rows.append(
                    base
                    | {
                        "event": "annotation_stroke",
                        "stroke_n": n,
                        "stroke_id": mark_id,
                        "kept": mark_id in live,
                        "start_ms": mt["start"],
                        "end_ms": mt["end"],
                    }
                )
        elif kind == "tlx":
            scores = {f"tlx_{k.replace('-', '_')}": v for k, v in answer.items()}
            rows.append(base | {"event": "tlx", "start_ms": a["startTime"], "end_ms": a["endTime"]} | scores)
        elif kind == "validation":
            rows.append(base | {"event": "validation", "start_ms": a["startTime"], "end_ms": a["endTime"]})
    df = pd.DataFrame(rows)
    df["source"] = "revisit"
    return df


def capture_procedures(rec_dir: str | Path) -> pd.DataFrame:
    """Calibrations and validations as Pupil Capture logged them, in Pupil time.

    Start/end are the first/last reference-marker sample, i.e. when the markers were on screen.
    """
    rows = []
    for n in load_pldata(rec_dir, "notify"):
        subject = n.get("subject")
        if subject == "calibration.setup.v2":
            refs, event = n["calib_data"]["ref_list"], "calibration"
        elif subject == "validation.data":
            refs, event = n["ref_list"], "validation"
        else:
            continue
        ts = [r["timestamp"] for r in refs]
        if ts:
            rows.append({"event": event, "start_pupil_s": min(ts), "end_pupil_s": max(ts), "source": "pupil_capture"})
    return pd.DataFrame(rows)


def screen_white_fraction(video: str | Path, fps: int = 60, size=(96, 79)) -> tuple[np.ndarray, np.ndarray]:
    """(t, white fraction of the content area) per frame of a screen recording, resampled to `fps`.

    The crop skips the browser chrome at the top, the prompt column on the left and the
    letterbox at the bottom, leaving the chart panel.
    """
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-an", "-vf", f"fps={fps},scale={w}:{h},format=gray"]
    raw = subprocess.run([*cmd, "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w)
    crop = frames[:, int(h * 0.20) : int(h * 0.80), int(w * 0.25) :]
    return np.arange(len(frames)) / fps, (crop > 235).mean((1, 2))


def crossings(t: np.ndarray, signal: np.ndarray, threshold: float) -> np.ndarray:
    """Times where `signal` crosses `threshold` (first sample on the new side)."""
    above = signal > threshold
    return t[np.flatnonzero(np.diff(above.astype(int)) != 0) + 1]


def transition_threshold(t: np.ndarray, signal: np.ndarray, chart_on: np.ndarray, lag_s: float = 0.0,
                         window_s: float = 1.0, guard_s: float = 0.5) -> tuple[float, float, float]:
    """Brightness threshold between the fixation slide and the chart slide, measured from the data.

    Median signal over `window_s` ending `guard_s` before each chart onset (fixation slide) and over
    `window_s` starting `guard_s` after it (chart slide); `lag_s` is a rough clock offset of the video.
    Returns (midpoint, fixation level, chart level). The guard keeps a few hundred ms of sync error
    out of both levels.
    """
    def level(a, b):
        vals = [np.nanmedian(signal[(t >= x + a) & (t < x + b)]) for x in chart_on + lag_s]
        return float(np.nanmedian(vals))

    lo = level(-guard_s - window_s, -guard_s)
    hi = level(guard_s, guard_s + window_s)
    return (lo + hi) / 2, lo, hi


def fit_offset(edges: np.ndarray, reference: np.ndarray, tol: float = 0.5, search: float | None = None) -> tuple:
    """Constant offset with `edges ≈ reference + offset`, plus per-reference residuals.

    The coarse offset is the candidate (edge - reference pair) that matches the most
    references within `tol`. `search` limits candidates to |offset| <= search. The final
    offset is the median of the matched differences. Returns (offset, DataFrame with
    `reference`, `edge`, `residual_s`, NaN where no edge was within `tol`).
    """
    candidates = (edges[:, None] - reference[None, :]).ravel()
    if search is not None:
        candidates = candidates[np.abs(candidates) <= search]
    if candidates.size == 0:
        raise ValueError("no candidate offsets")

    def matches(off):
        d = np.abs(edges[None, :] - (reference[:, None] + off))
        return (d.min(1) <= tol).sum()

    coarse = max(np.unique(candidates.round(3)), key=matches)
    nearest = edges[np.abs(edges[None, :] - (reference[:, None] + coarse)).argmin(1)]
    diff = nearest - reference
    ok = np.abs(diff - coarse) <= tol
    offset = float(np.median(diff[ok]))
    res = pd.DataFrame({"reference": reference, "edge": np.where(ok, nearest, np.nan)})
    res["residual_s"] = res.edge - (res.reference + offset)
    return offset, res


def chart_transitions(events: pd.DataFrame) -> pd.Series:
    """Unix-ms times when a chart panel appears (graph start) or disappears (graph end)."""
    g = events[events.event == "graph"]
    return pd.concat(
        [
            g.start_ms.rename("t").to_frame().assign(kind="chart_on"),
            g.end_ms.rename("t").to_frame().assign(kind="chart_off"),
        ]
    ).sort_values("t")


def build_timeline(study_json: str | Path, rec_dir: str | Path, screen_offset_s: float | None = None) -> pd.DataFrame:
    """All study + Pupil Capture events with start/end on every clock.

    `screen_offset_s`: Pupil time of screen-recording t=0 (from `fit_offset`); omit to skip
    the screen columns.
    """
    rec_dir = Path(rec_dir)
    unix_to_pupil = unix_to_pupil_offset(rec_dir)
    ev = study_events(load_answers(study_json))
    ev["start_pupil_s"] = ev.start_ms / 1000 + unix_to_pupil
    ev["end_pupil_s"] = ev.end_ms / 1000 + unix_to_pupil

    cap = capture_procedures(rec_dir)
    if not cap.empty:
        cap["start_ms"] = (cap.start_pupil_s - unix_to_pupil) * 1000
        cap["end_ms"] = (cap.end_pupil_s - unix_to_pupil) * 1000
        # attach each Capture procedure to the reVISit slide it ran during, if any
        val = ev[ev.event == "validation"]
        for i, c in cap.iterrows():
            hit = val[(val.start_pupil_s <= c.start_pupil_s) & (val.end_pupil_s >= c.end_pupil_s)]
            if len(hit):
                h = hit.iloc[0]
                cap.loc[i, ["block_index", "block", "component"]] = [h.block_index, h.block, h.component]
                cap.loc[i, "event"] = "validation_markers"
            else:
                cap.loc[i, ["block_index", "block"]] = [0, "pre-study"]
        ev = pd.concat([ev, cap], ignore_index=True)

    world_ts = load_frame_lookup(rec_dir).timestamp.to_numpy()
    for side in ("start", "end"):
        p = ev[f"{side}_pupil_s"]
        ev[f"{side}_world_s"] = p - world_ts[0]
        frame = timestamps_to_frames(p, world_ts)
        ev[f"{side}_world_frame"] = np.where((p >= world_ts[0]) & (p <= world_ts[-1]), frame, -1)
        if screen_offset_s is not None:
            ev[f"{side}_screen_s"] = p - screen_offset_s
    ev["duration_s"] = ev.end_pupil_s - ev.start_pupil_s
    ev["in_recording"] = (ev.start_pupil_s >= world_ts[0]) & (ev.end_pupil_s <= world_ts[-1])
    ev = ev.sort_values(["start_pupil_s", "end_pupil_s"]).reset_index(drop=True)
    cols = [
        "block_index", "block", "task", "event", "stroke_n", "stroke_id", "kept", "answer", "correct_answer",
        "is_correct", "start_ms", "end_ms", "start_pupil_s", "end_pupil_s", "duration_s", "start_world_s",
        "end_world_s", "start_world_frame", "end_world_frame", "start_screen_s", "end_screen_s", "in_recording",
        "component", "source", *TLX_COLUMNS,
    ]  # fmt: skip
    ev = ev[[c for c in cols if c in ev.columns]].astype({"block_index": "Int64"})
    if "stroke_n" in ev:
        ev["stroke_n"] = ev.stroke_n.astype("Int64")
    return ev.rename(columns={"start_ms": "start_unix_ms", "end_ms": "end_unix_ms"})
