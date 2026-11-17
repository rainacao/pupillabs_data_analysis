"""Event timeline CSV for one participant: study events on the Pupil / world / screen clocks.

    uv run python scripts/event_timeline.py recordings/p1

Expects in the participant folder: one Pupil Core recording folder (has info.player.json),
the reVISit participant JSON, and `<participant>_screen.mov`. Writes
`<participant>_event_timeline.csv` and `<participant>_sync_check.csv` (per-transition
residuals of the clock alignment against the world video and the screen recording).
If the study asked a NASA-TLX after each chart, also `<participant>_tlx.csv` (one row per chart).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pupil_analysis.io import load_frame_lookup
from pupil_analysis.marks import unix_to_pupil_offset
from pupil_analysis.provenance import fit_summary, write_record
from pupil_analysis.timeline import (
    TLX_COLUMNS,
    build_timeline,
    chart_transitions,
    crossings,
    fit_offset,
    load_answers,
    screen_white_fraction,
    study_events,
    transition_threshold,
)
from pupil_analysis.video import white_fraction

# world white fraction: threshold measured per participant (`transition_threshold`), since camera
# exposure differs (p1: fixation ~0.09 / chart ~0.12; p2: ~0.002 / ~0.045)
SCREEN_THRESHOLD = 0.2  # screen white fraction: fixation slide 0, chart slide 0.38-0.65


def cached(path: Path, compute):
    if path.exists():
        return np.load(path)
    arr = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("participant_dir", type=Path)
    args = ap.parse_args()
    pdir: Path = args.participant_dir
    pid = pdir.name
    rec = next(p.parent for p in pdir.glob("*/info.player.json"))
    study = next(pdir.glob("VLAT_*.json"))
    screen = pdir / f"{pid}_screen.mov"
    cache = pdir / ".cache"
    print(f"recording {rec}\nstudy     {study.name}\nscreen    {screen.name}")

    unix_to_pupil = unix_to_pupil_offset(rec)
    trans = chart_transitions(study_events(load_answers(study)))
    ref = trans.t.to_numpy() / 1000 + unix_to_pupil  # Pupil time of each chart on/off

    # 1) check unix -> Pupil conversion against slide changes seen by the world camera
    world_t = load_frame_lookup(rec).timestamp.to_numpy()
    world_sig = cached(cache / "world_white.npy", lambda: white_fraction(rec, step=1, smooth=5))
    on = ref[trans.kind.to_numpy() == "chart_on"]
    world_thr, world_lo, world_hi = transition_threshold(world_t, world_sig, on)
    print(f"world white fraction: fixation {world_lo:.3f}, chart {world_hi:.3f} -> threshold {world_thr:.3f}")
    world_lag, world_res = fit_offset(crossings(world_t, world_sig, world_thr), ref, tol=0.5, search=2.0)

    # 2) place the screen recording on the Pupil clock
    st = cached(cache / "screen_white.npy", lambda: np.stack(screen_white_fraction(screen)))
    screen_off, screen_res = fit_offset(crossings(st[0], st[1], SCREEN_THRESHOLD), ref, tol=0.2)
    screen_zero = -screen_off  # Pupil time at screen-recording t=0

    sync = pd.concat(
        [
            r.assign(clock=name, kind=trans.kind.to_numpy(), fitted_offset_s=off)
            for name, off, r in (("world_video", world_lag, world_res), ("screen_recording", screen_off, screen_res))
        ]
    ).rename(columns={"reference": "revisit_pupil_s", "edge": "detected_s"})
    sync.to_csv(pdir / f"{pid}_sync_check.csv", index=False)
    for name, r, off in (("world video", world_res, world_lag), ("screen recording", screen_res, screen_off)):
        res = r.residual_s.dropna() * 1000
        print(
            f"{name}: offset {off:+.3f} s, matched {len(res)}/{len(r)} transitions, "
            f"residual median {res.median():+.0f} ms, |max| {res.abs().max():.0f} ms"
        )

    write_record(
        pdir / f"{pid}_sync_params.json",
        {
            "inputs": {"recording": rec, "study_json": study, "screen_recording": screen},
            "unix_to_pupil": {
                "offset_s": unix_to_pupil,
                "how": "automatic: info.player.json start_time_synced_s - start_time_system_s "
                "(browser and Pupil Capture on the same Mac)",
            },
            "world_video_lag": {
                "offset_s": world_lag,
                "meaning": "world camera sees a slide change this long after reVISit logs it",
                "how": "automatic: median offset of chart on/off brightness edges in the world video",
                "fit": fit_summary(world_res.residual_s),
                "parameters": {"white_fraction_threshold": world_thr,
                               "threshold_how": "automatic: midpoint of the median fixation-slide level "
                               f"({world_lo:.4f}) and chart-slide level ({world_hi:.4f}) around chart onsets",
                               "match_tolerance_s": 0.5,
                               "search_s": 2.0, "smoothing_frames": 5},
            },
            "screen_recording": {
                "offset_s": screen_off,
                "meaning": "screen-recording time = Pupil time + offset_s",
                "pupil_time_at_screen_t0": screen_zero,
                "how": "automatic: median offset of chart on/off brightness edges in the screen recording",
                "fit": fit_summary(screen_res.residual_s),
                "parameters": {"white_fraction_threshold": SCREEN_THRESHOLD, "match_tolerance_s": 0.2,
                               "decode_fps": 60, "crop": "rows 20-80%, columns 25-100%"},
            },
            "per_transition_residuals": f"{pid}_sync_check.csv",
        },
    )  # fmt: skip

    timeline = build_timeline(study, rec, screen_offset_s=screen_zero)
    out = pdir / f"{pid}_event_timeline.csv"
    timeline.to_csv(out, index=False)
    print(f"wrote {out} ({len(timeline)} events)")
    tlx = timeline[timeline.event == "tlx"]
    if len(tlx):
        cols = ["block_index", "block", "task", "start_unix_ms", "end_unix_ms", "start_pupil_s", "end_pupil_s",
                "duration_s", "start_screen_s", "end_screen_s", "in_recording", *TLX_COLUMNS]  # fmt: skip
        tlx_out = pdir / f"{pid}_tlx.csv"
        tlx[cols].to_csv(tlx_out, index=False)
        print(f"wrote {tlx_out} ({len(tlx)} NASA-TLX responses)")
    print(timeline.groupby(["block_index", "block", "event"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
