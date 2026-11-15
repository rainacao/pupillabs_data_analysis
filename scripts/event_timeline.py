"""Event timeline CSV for one participant: study events on the Pupil / world / screen clocks.

    uv run python scripts/event_timeline.py recordings/p1

Expects in the participant folder: one Pupil Core recording folder (has info.player.json),
the reVISit participant JSON, and `<participant>_screen.mov`. Writes
`<participant>_event_timeline.csv` and `<participant>_sync_check.csv` (per-transition
residuals of the clock alignment against the world video and the screen recording).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pupil_analysis.io import load_frame_lookup
from pupil_analysis.marks import unix_to_pupil_offset
from pupil_analysis.timeline import (
    build_timeline,
    chart_transitions,
    crossings,
    fit_offset,
    load_answers,
    screen_white_fraction,
    study_events,
)
from pupil_analysis.video import white_fraction

WORLD_THRESHOLD = 0.11  # world white fraction: fixation slide ~0.09, chart slide ~0.12-0.13
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
    study = next(pdir.glob("*.json"))
    screen = pdir / f"{pid}_screen.mov"
    cache = pdir / ".cache"
    print(f"recording {rec}\nstudy     {study.name}\nscreen    {screen.name}")

    unix_to_pupil = unix_to_pupil_offset(rec)
    trans = chart_transitions(study_events(load_answers(study)))
    ref = trans.t.to_numpy() / 1000 + unix_to_pupil  # Pupil time of each chart on/off

    # 1) check unix -> Pupil conversion against slide changes seen by the world camera
    world_t = load_frame_lookup(rec).timestamp.to_numpy()
    world_sig = cached(cache / "world_white.npy", lambda: white_fraction(rec, step=1, smooth=5))
    world_lag, world_res = fit_offset(crossings(world_t, world_sig, WORLD_THRESHOLD), ref, tol=0.5, search=2.0)

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

    timeline = build_timeline(study, rec, screen_offset_s=screen_zero)
    out = pdir / f"{pid}_event_timeline.csv"
    timeline.to_csv(out, index=False)
    print(f"wrote {out} ({len(timeline)} events)")
    print(timeline.groupby(["block_index", "block", "event"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
