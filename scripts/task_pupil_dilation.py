"""Per-task pupil dilation for one participant, baselined on the 5 s fixation cross before each task.

    uv run python scripts/event_timeline.py recordings/p1      # first, if the timeline CSV is missing
    uv run python scripts/task_pupil_dilation.py recordings/p1

Reads `<pid>_event_timeline.csv` and writes `<pid>_pupil_dilation.csv` (one row per task) and
`<pid>_pupil_dilation.png` (traces, for checking).

Primary measure: 2D pupil diameter (ellipse major axis, eye-camera px), reported as px and as %
change from each eye's own baseline. pye3d's 3D diameter (mm) is included as `mm3d_*` columns for
reference only. On p1 the eye0 3D model refit repeatedly (eyeball centre wandered 29-55 mm from the
camera), producing step artifacts; within-trial agreement between the eyes was median r=0.70 (2D)
vs 0.44 (3D), RMS difference 6.1% vs 12.2% of baseline.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pupil_analysis.pupil import binned_trace, load_pupil_diameter, window_dilation

# 3D (reference only): pye3d puts the two eyes at different absolute scales (p1: eye0 median
# 1.7 mm, eye1 2.7 mm), so the default 1.5 mm plausibility floor would truncate a quarter of
# eye0's samples. Keep only a gross-error floor; blinks are removed by confidence + `clean_eye`.
VALID_MM = (0.5, 9.0)
MIN_CONFIDENCE = 0.8


def task_windows(timeline: pd.DataFrame) -> pd.DataFrame:
    """Pair each graph with the fixation cross of the same task: baseline = cross shown -> hidden."""
    cross = timeline[timeline.event == "fixation_cross"].set_index(["block", "task"])
    graph = timeline[timeline.event == "graph"]
    rows = []
    for order, g in enumerate(graph.to_dict("records"), 1):
        c = cross.loc[(g["block"], g["task"])]
        rows.append(
            {
                "order": order,
                "block_index": g["block_index"],
                "block": g["block"],
                "task": g["task"],
                "is_correct": g["is_correct"],
                "baseline_start": c["start_pupil_s"],
                "baseline_end": c["end_pupil_s"],
                "task_start": g["start_pupil_s"],
                "task_end": g["end_pupil_s"],
                "task_duration_s": g["end_pupil_s"] - g["start_pupil_s"],
                "n_strokes": int(((timeline.event == "annotation_stroke") & (timeline.task == g["task"])
                                  & (timeline.block == g["block"])).sum()),
            }
        )  # fmt: skip
    return pd.DataFrame(rows)


def plot(pupil: pd.DataFrame, result: pd.DataFrame):
    """Per task: each eye's % change from its baseline over fixation cross + task, with the mean and peak."""
    fig, axes = plt.subplots(3, 6, figsize=(20, 9), sharey=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, (_, block) in enumerate(result.groupby("block_index")):
        for j, r in enumerate(block.to_dict("records")):
            ax = axes[i, j]
            ax.set_visible(True)
            base = pd.Series({0: r["baseline_eye0_px"], 1: r["baseline_eye1_px"]})
            tr = binned_trace(pupil, r["baseline_start"], r["task_end"], 0.05, "diameter_px")
            tr = 100 * tr.sub(base, axis=1).div(base, axis=1)
            t = tr.index + (r["baseline_start"] - r["task_start"])
            for eye in (0, 1):
                if eye in tr:
                    ax.plot(t, tr[eye], lw=0.6, alpha=0.6, label=f"eye{eye}")
            ax.axvline(0, color="k", lw=0.8)
            ax.axhline(0, color="grey", lw=0.5)
            ax.hlines(r["mean_change_pct"], 0, r["task_duration_s"], color="C3", label="task mean")
            ax.plot(r["peak_time_s"], r["peak_dilation_pct"], "v", color="C3", label="peak")
            ax.set_title(f"{r['block']} / {r['task']}", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("s from chart onset (negative = fixation cross)")
    fig.supylabel("2D diameter, % change from baseline")
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("participant_dir", type=Path)
    args = ap.parse_args()
    pdir: Path = args.participant_dir
    pid = pdir.name
    rec = next(p.parent for p in pdir.glob("*/info.player.json"))
    timeline = pd.read_csv(pdir / f"{pid}_event_timeline.csv")

    windows = task_windows(timeline)
    px = load_pupil_diameter(rec, min_confidence=MIN_CONFIDENCE, measure="2d")
    mm = load_pupil_diameter(rec, min_confidence=MIN_CONFIDENCE, valid_mm=VALID_MM, measure="3d")
    result = window_dilation(px, windows, col="diameter_px")
    ref = window_dilation(mm, windows, col="diameter_mm", prefix="mm3d_")
    keep = ["mm3d_baseline_eye0_mm", "mm3d_baseline_eye1_mm", "mm3d_mean_change_mm", "mm3d_mean_change_pct",
            "mm3d_peak_dilation_mm", "mm3d_peak_dilation_pct", "mm3d_peak_time_s"]  # fmt: skip
    result = pd.concat([result, ref[keep]], axis=1)
    out = pdir / f"{pid}_pupil_dilation.csv"
    result.to_csv(out, index=False)
    plot(px, result).savefig(pdir / f"{pid}_pupil_dilation.png", dpi=110)

    cols = ["block", "task", "baseline_eye0_px", "baseline_eye1_px", "mean_change_px", "mean_change_pct",
            "peak_dilation_px", "peak_dilation_pct", "peak_time_s", "task_coverage", "mm3d_mean_change_pct"]  # fmt: skip
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(result[cols].to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
