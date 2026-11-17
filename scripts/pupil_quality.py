"""How much pupil data survives cleaning, per task window and eye.

    uv run python scripts/task_pupil_dilation.py recordings/p2   # first: defines the windows
    uv run python scripts/pupil_quality.py recordings/p2

For each task's baseline (fixation cross) and chart window, per eye:
- `rate_hz`: pupil samples per second the eye camera delivered (all confidences).
- `conf_ok`: fraction with confidence >= 0.8 and a physically possible 2D diameter.
- `kept_clean_eye`: fraction left after the default cleaning (confidence + `clean_eye`).
- `kept_pupil_labs`: fraction left after confidence + Pupil Player's blinks.
- `bins_clean_eye`: fraction of 50 ms bins with at least one sample after the default cleaning; this is
  what the dilation traces and peaks are built from.
- `blinks`, `blink_s`: Pupil Player blinks overlapping the window, and the time they cover (s).

Writes `<pid>_pupil_quality.csv` (one row per task x window x eye) and prints per-eye summaries.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pupil_analysis.io import load_pldata
from pupil_analysis.pupil import load_blinks, load_pupil_diameter

MIN_CONFIDENCE = 0.8
BIN_S = 0.05


def raw_pupil(rec: Path) -> pd.DataFrame:
    """Every pupil datum the dilation analysis reads (one per eye frame), before any filtering."""
    offline = rec / "offline_data" / "offline_pupil.pldata"
    rows = load_pldata(offline.parent, offline.stem) if offline.exists() else load_pldata(rec, "pupil")
    rows = [r for r in rows if "diameter_3d" in r]
    return pd.DataFrame({"timestamp": [r["timestamp"] for r in rows], "eye": [r["id"] for r in rows]})


def count(df: pd.DataFrame, eye: int, t0: float, t1: float) -> int:
    return int(((df.eye == eye) & (df.timestamp >= t0) & (df.timestamp < t1)).sum())


def bin_coverage(df: pd.DataFrame, eye: int, t0: float, t1: float) -> float:
    t = df.timestamp[(df.eye == eye) & (df.timestamp >= t0) & (df.timestamp < t1)].to_numpy()
    n_bins = max(1, int(np.ceil((t1 - t0) / BIN_S)))
    return len(np.unique(((t - t0) // BIN_S).astype(int))) / n_bins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("participant_dir", type=Path)
    args = ap.parse_args()
    pdir: Path = args.participant_dir
    pid = pdir.name
    rec = next(p.parent for p in pdir.glob("*/info.player.json"))
    windows = pd.read_csv(pdir / f"{pid}_pupil_dilation.csv")

    raw = raw_pupil(rec)
    conf = load_pupil_diameter(rec, min_confidence=MIN_CONFIDENCE, measure="2d", cleaning="none")
    clean = load_pupil_diameter(rec, min_confidence=MIN_CONFIDENCE, measure="2d", cleaning="kret")
    pl = load_pupil_diameter(rec, min_confidence=MIN_CONFIDENCE, measure="2d", cleaning="pupil_labs")
    blinks = load_blinks(rec)

    rows = []
    for w in windows.to_dict("records"):
        for name, t0, t1 in (("baseline", w["baseline_start"], w["baseline_end"]),
                             ("chart", w["task_start"], w["task_end"])):  # fmt: skip
            b = blinks[(blinks.end > t0) & (blinks.start < t1)]
            blink_s = float((np.minimum(b.end, t1) - np.maximum(b.start, t0)).sum())
            for eye in (0, 1):
                n = count(raw, eye, t0, t1)
                rows.append({
                    "block": w["block"], "task": w["task"], "window": name, "eye": eye, "duration_s": t1 - t0,
                    "rate_hz": n / (t1 - t0),
                    "conf_ok": count(conf, eye, t0, t1) / n if n else np.nan,
                    "kept_clean_eye": count(clean, eye, t0, t1) / n if n else np.nan,
                    "kept_pupil_labs": count(pl, eye, t0, t1) / n if n else np.nan,
                    "bins_clean_eye": bin_coverage(clean, eye, t0, t1),
                    "blinks": len(b), "blink_s": blink_s, "blink_frac": blink_s / (t1 - t0),
                })  # fmt: skip
    q = pd.DataFrame(rows)
    out = pdir / f"{pid}_pupil_quality.csv"
    q.to_csv(out, index=False)

    cols = ["rate_hz", "conf_ok", "kept_clean_eye", "kept_pupil_labs", "bins_clean_eye"]
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200):
        print("median over tasks, per window and eye:")
        print(q.groupby(["window", "eye"])[cols].median().to_string())
        print("\nper task (chart window), default cleaning: fraction of samples kept / of 50 ms bins with data")
        piv = q[q.window == "chart"].pivot_table(index=["block", "task"], columns="eye",
                                                 values=["kept_clean_eye", "bins_clean_eye"])  # fmt: skip
        per_task = q[(q.window == "chart") & (q.eye == 0)].set_index(["block", "task"])
        piv[("blinks", "")] = per_task.blinks
        piv[("blinks_per_min", "")] = 60 * per_task.blinks / per_task.duration_s
        print(piv.to_string())
        tot = q[q.eye == 0]
        print(f"\nPlayer blinks: {tot.blinks.sum()} in {tot.duration_s.sum():.0f} s of task windows "
              f"({60 * tot.blinks.sum() / tot.duration_s.sum():.1f}/min, {tot.blink_s.sum() / tot.duration_s.sum():.1%} of the time)")  # fmt: skip
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
