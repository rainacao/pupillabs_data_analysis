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
BIN_S = 0.05  # trace bins (median per eye)
PEAK_SMOOTH_S = 0.25  # rolling mean before taking the peak

# Task selection, so the same task lines up across chart types.
# Stackarea's absolute-value question counts as "value"; its relative variants are not analysed.
TASK_ALIASES = {("stackarea", "value-absolute"): "value"}
DROP_TASKS = {("stackarea", "value-relative"), ("stackarea", "comparisons-relative")}


def task_windows(timeline: pd.DataFrame) -> pd.DataFrame:
    """Pair each graph with the fixation cross of the same task: baseline = cross shown -> hidden.

    Tasks in DROP_TASKS are skipped and TASK_ALIASES renames tasks; `source_task` keeps the
    study's own name. `order` is the chart's position among all charts shown, dropped ones included.
    """
    cross = timeline[timeline.event == "fixation_cross"].set_index(["block", "task"])
    graph = timeline[timeline.event == "graph"]
    rows = []
    for order, g in enumerate(graph.to_dict("records"), 1):
        key = (g["block"], g["task"])
        if key in DROP_TASKS:
            continue
        c = cross.loc[key]
        rows.append(
            {
                "order": order,
                "block_index": g["block_index"],
                "block": g["block"],
                "task": TASK_ALIASES.get(key, g["task"]),
                "source_task": g["task"],
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


# Column order of the plot grid: related tasks side by side. Tasks not listed go at the end.
TASK_ORDER = ["value", "comparisons", "trends", "extremum", "range"]


def plot(pupil: pd.DataFrame, result: pd.DataFrame):
    """Grid of rows = chart type (presentation order), columns = task (same task in every row).

    Each cell: each eye's % change from its baseline over fixation cross + task, with the mean and
    peak. Cells for a task that chart type didn't have are left empty.
    """
    present = list(result.task.unique())
    tasks = [t for t in TASK_ORDER if t in present] + [t for t in present if t not in TASK_ORDER]
    blocks = result.drop_duplicates("block_index").sort_values("block_index").block.tolist()
    fig, axes = plt.subplots(len(blocks), len(tasks), figsize=(3.2 * len(tasks), 3 * len(blocks)), sharey=True)
    axes = axes.reshape(len(blocks), len(tasks))
    for j, task in enumerate(tasks):
        axes[0, j].set_title(task, fontsize=11, fontweight="bold")
    for i, block in enumerate(blocks):
        axes[i, 0].set_ylabel(f"$\\bf{{{block}}}$\n% change from baseline", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.text(0.5, 0.5, "not in block", transform=ax.transAxes, ha="center", va="center", color="0.6", fontsize=8)
        for side in ax.spines.values():
            side.set_color("0.85")
    for r in result.to_dict("records"):
        ax = axes[blocks.index(r["block"]), tasks.index(r["task"])]
        ax.texts[0].remove()
        ax.set_xticks([-5, 0, 5, 10, 15, 20, 25])
        ax.set_xlim(r["baseline_start"] - r["task_start"], r["task_duration_s"])
        for side in ax.spines.values():
            side.set_color("black")
        base = pd.Series({0: r["baseline_eye0_px"], 1: r["baseline_eye1_px"]})
        # binned from the cross onset and from the chart onset separately, so the chart part
        # uses the same bins as window_dilation and the peak marker sits on the average line
        for t0, t1 in ((r["baseline_start"], r["task_start"]), (r["task_start"], r["task_end"])):
            tr = binned_trace(pupil, t0, t1, BIN_S, "diameter_px").reindex(columns=[0, 1])
            tr = 100 * tr.sub(base, axis=1).div(base, axis=1)
            t = tr.index + BIN_S / 2 + (t0 - r["task_start"])  # bin centres, s from chart onset
            avg = tr.mean(axis=1).rolling(round(PEAK_SMOOTH_S / BIN_S), center=True, min_periods=1).mean()
            first = t0 == r["baseline_start"]
            for eye, color in ((0, "C0"), (1, "C1")):
                ax.plot(t, tr[eye], lw=0.5, alpha=0.35, color=color, label=f"eye{eye}" if first else None)
            ax.plot(t, avg, lw=1.3, color="k", label="mean of eyes" if first else None)
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.hlines(r["mean_change_pct"], 0, r["task_duration_s"], color="C3", label="task mean")
        ax.plot(r["peak_time_s"], r["peak_dilation_pct"], "v", color="C3", label="peak")
        ax.text(0.98, 0.04, f"mean {r['mean_change_pct']:+.1f}%  peak {r['peak_dilation_pct']:+.1f}%",
                transform=ax.transAxes, ha="right", fontsize=7)  # fmt: skip
    # legend goes in an empty cell if there is one, otherwise on the first task
    first = result.iloc[0]
    handles, labels = axes[blocks.index(first.block), tasks.index(first.task)].get_legend_handles_labels()
    empty = [ax for ax in axes.flat if ax.texts and ax.texts[0].get_text() == "not in block"]
    if empty:
        empty[0].legend(handles, labels, fontsize=8, loc="upper center", frameon=False)
    else:
        axes[0, 0].legend(handles, labels, fontsize=7, loc="upper left")
    fig.supxlabel("s from chart onset (negative = fixation cross)")
    fig.suptitle("2D pupil diameter, % change from each eye's fixation-cross baseline", fontsize=12)
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
    result = window_dilation(px, windows, col="diameter_px", bin_s=BIN_S, peak_smooth_s=PEAK_SMOOTH_S)
    ref = window_dilation(mm, windows, col="diameter_mm", bin_s=BIN_S, peak_smooth_s=PEAK_SMOOTH_S, prefix="mm3d_")
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
