"""NASA-TLX ratings per task, compared across participants.

    uv run python scripts/tlx_plot.py recordings/p2 recordings/p3

Reads the ratings straight from each participant's reVISit JSON (no Pupil recording needed), so it
also works for participants whose event timeline has not been built. One panel per TLX scale; rows are
the chart/task pairs grouped by chart type; one marker per participant. Writes `tlx_<pids>.png` and the
long-format ratings `tlx_<pids>.csv` next to the participant folders.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pupil_analysis.timeline import TLX_COLUMNS, TLX_SCALES, load_answers, study_events

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
MARKERS = ["o", "D", "s", "^"]  # secondary encoding so identity is not color-alone
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLOCK_ORDER = ["area", "stackarea", "line"]
SCALE_LABELS = {
    "mental-demand": "Mental demand",
    "physical-demand": "Physical demand",
    "temporal-demand": "Temporal demand",
    "performance": "Performance (scale direction unverified)",
    "effort": "Effort",
    "frustration": "Frustration",
}


def load_tlx(pdir: Path) -> pd.DataFrame:
    """One row per chart: participant, block, task and the six ratings, from the reVISit JSON."""
    study = next(pdir.glob("VLAT_*.json"))
    ev = study_events(load_answers(study))
    tlx = ev[ev.event == "tlx"][["block", "task", *TLX_COLUMNS]].copy()
    tlx.insert(0, "participant", pdir.name)
    return tlx.reset_index(drop=True)


def plot_tlx(df: pd.DataFrame) -> plt.Figure:
    pids = list(dict.fromkeys(df.participant))
    rows = df[["block", "task"]].drop_duplicates()
    rows = rows.assign(b=rows.block.map(BLOCK_ORDER.index)).sort_values(["b", "task"])
    keys = list(zip(rows.block, rows.task))
    y = {k: i for i, k in enumerate(keys)}
    # small vertical offset per participant so equal ratings don't hide each other
    dodge = {pid: (j - (len(pids) - 1) / 2) * 0.34 for j, pid in enumerate(pids)}

    fig, axes = plt.subplots(2, 3, figsize=(13, 9), sharex=True, sharey=True)
    for ax, scale, col in zip(axes.flat, TLX_SCALES, TLX_COLUMNS):
        # separators between chart types
        for i in range(1, len(keys)):
            if keys[i][0] != keys[i - 1][0]:
                ax.axhline(i - 0.5, color=GRID, linewidth=1)
        # thin connector between participants on the same task
        for _, g in df.groupby(["block", "task"]):
            if len(g) > 1:
                ys = [y[(b, t)] + dodge[p] for b, t, p in zip(g.block, g.task, g.participant)]
                ax.plot(g[col], ys, color=GRID, linewidth=2, zorder=1)
        for j, pid in enumerate(pids):
            d = df[df.participant == pid]
            ys = [y[k] + dodge[pid] for k in zip(d.block, d.task)]
            ax.scatter(d[col], ys, s=42, color=COLORS[j], marker=MARKERS[j], edgecolor="white",
                       linewidth=1.5, zorder=3, label=pid)  # fmt: skip
        ax.set_title(SCALE_LABELS[scale], loc="left", fontsize=11, color=INK)
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax0 = axes[0, 0]
    ax0.set_yticks(range(len(keys)))
    ax0.set_yticklabels([f"{b} · {t}" for b, t in keys])
    ax0.set_ylim(len(keys) - 0.5, -0.5)
    for ax in axes[1]:
        ax.set_xlabel("rating (0–100)", color=MUTED, fontsize=9)
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(pids), frameon=False, fontsize=10)
    fig.suptitle("NASA-TLX ratings per task", x=0.01, ha="left", fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("participant_dirs", type=Path, nargs="+")
    args = ap.parse_args()
    df = pd.concat([load_tlx(p) for p in args.participant_dirs], ignore_index=True)
    stem = "tlx_" + "_".join(p.name for p in args.participant_dirs)
    out_dir = args.participant_dirs[0].parent
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    fig = plot_tlx(df)
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    print(f"wrote {out_dir / stem}.png and .csv ({len(df)} ratings rows)")


if __name__ == "__main__":
    main()
