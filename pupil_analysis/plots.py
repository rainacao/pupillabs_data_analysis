"""Histograms of fixation durations and inter-fixation gaps per chart."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from pupil_analysis.events import fixations_in_interval, saccades_between

FIXATION_COLOR = "#2a78d6"
GAP_COLOR = "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def binned(values: np.ndarray, bin_ms: float, max_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Left bin edges and counts for `bin_ms` bins from 0 to `max_ms`, plus a final overflow bin for >= `max_ms`."""
    edges = np.arange(0, max_ms + bin_ms, bin_ms)  # last edge == max_ms starts the overflow bin
    counts = np.bincount(np.minimum(np.asarray(values) // bin_ms, len(edges) - 1).astype(int), minlength=len(edges))
    return edges, counts


def _panel(ax: Axes, values: np.ndarray, bin_ms: float, max_ms: float, color: str) -> None:
    edges, counts = binned(values, bin_ms, max_ms)
    ax.bar(edges, counts, width=bin_ms * 0.9, align="edge", color=color, linewidth=0)
    if values.size:
        med = float(np.median(values))
        ax.axvline(min(med, max_ms), color=INK, linewidth=1, linestyle=(0, (3, 2)))
        ax.text(
            0.98,
            0.95,
            f"n = {values.size}\nmedian {med:.0f} ms",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color=MUTED,
        )
    ticks = np.arange(0, max_ms, 100 if max_ms <= 500 else 200)
    ax.set_xticks(np.append(ticks, max_ms + bin_ms / 2))
    ax.set_xticklabels([f"{t:g}" for t in ticks] + [f"≥{max_ms:g}"])
    ax.set_xlim(0, max_ms + bin_ms)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def plot_event_histograms(
    fixations: pd.DataFrame,
    intervals: dict[str, tuple[int, int]],
    frame_ts: np.ndarray,
    bin_ms: float = 20,
    fixation_max_ms: float = 320,
    gap_max_ms: float = 1000,
) -> Figure:
    """One row per interval: fixation durations (left) and gaps between consecutive fixations (right).

    Values at or above `*_max_ms` are collected in each panel's last bar.
    """
    fig, axes = plt.subplots(
        len(intervals), 2, figsize=(11, 2.6 * len(intervals)), sharex="col", sharey="col", squeeze=False
    )
    for row, (name, (start, end)) in enumerate(intervals.items()):
        fx = fixations_in_interval(fixations, start, end, frame_ts)
        gaps = saccades_between(fx).gap_ms.to_numpy()
        _panel(axes[row, 0], fx.duration.to_numpy(), bin_ms, fixation_max_ms, FIXATION_COLOR)
        _panel(axes[row, 1], gaps, bin_ms, gap_max_ms, GAP_COLOR)
        axes[row, 0].set_ylabel(f"{name}\ncount", color=INK, fontsize=10)

    axes[0, 0].set_title("Fixation duration", loc="left", color=INK, fontsize=11)
    axes[0, 1].set_title("Gap between consecutive fixations (saccade proxy)", loc="left", color=INK, fontsize=11)
    axes[-1, 0].set_xlabel(f"duration (ms), {bin_ms:g} ms bins", color=MUTED, fontsize=9)
    axes[-1, 1].set_xlabel(f"gap (ms), {bin_ms:g} ms bins", color=MUTED, fontsize=9)
    fig.tight_layout()
    return fig


EYE_COLORS = {0: FIXATION_COLOR, 1: GAP_COLOR}  # categorical slots 1 and 2
SHADE = "#efeeea"


def _style(ax: Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def plot_pupil_timeline(pupil: pd.DataFrame, trials, frame_ts: np.ndarray, smooth_s: float = 0.25) -> Figure:
    """Whole-session diameter per eye, fixation slides shaded and stimuli labelled."""
    t0 = frame_ts[trials[0].fix_start] - 2
    t1 = frame_ts[trials[-1].end] + 2
    fig, ax = plt.subplots(figsize=(15, 4.2))
    for tr in trials:
        ax.axvspan(frame_ts[tr.fix_start] - t0, frame_ts[tr.onset] - t0, color=SHADE, linewidth=0)
        mid = (frame_ts[tr.onset] + frame_ts[tr.end]) / 2 - t0
        ax.text(mid, 1.01, tr.name.split("_", 1)[0], transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, color=MUTED)
    win = pupil[(pupil.timestamp >= t0) & (pupil.timestamp <= t1)]
    for eye, grp in win.groupby("eye"):
        s = grp.set_index(pd.to_timedelta(grp.timestamp - t0, unit="s")).diameter_mm
        s = s.rolling(f"{int(smooth_s * 1000)}ms", center=True).median()
        ax.plot(s.index.total_seconds(), s.to_numpy(), color=EYE_COLORS[eye], linewidth=1.2, label=f"eye {eye}")
    ax.set_xlim(0, t1 - t0)
    ax.set_xlabel("time (s); grey = fixation slide, numbers = chart", color=MUTED, fontsize=9)
    ax.set_ylabel("pupil diameter (mm, pye3d)", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1, 1.12), ncol=2)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_trial_traces(traces: pd.DataFrame, ncols: int = 4) -> Figure:
    """Small multiples: change from the fixation baseline, per chart, both eyes plus their mean."""
    names = list(dict.fromkeys(traces.trial))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows), sharey=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        d = traces[traces.trial == name]
        ax.axvspan(d.t.min(), 0, color=SHADE, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        for eye in (0, 1):
            ax.plot(d.t, d[f"change_eye{eye}"], color=EYE_COLORS[eye], linewidth=0.8, alpha=0.6)
        ax.plot(d.t, d.change_mm, color=INK, linewidth=1.6)
        stim = d[d.t >= 0].change_mm.mean()
        ax.set_title(f"{name.replace('_', ' ', 1).replace('_', ' ')}", loc="left", fontsize=10, color=INK)
        ax.text(0.98, 0.04, f"mean {stim:+.2f} mm", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color=MUTED)
        _style(ax)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ diameter (mm)", color=INK, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("s from chart onset", color=MUTED, fontsize=9)
    handles = [plt.Line2D([], [], color=EYE_COLORS[0], lw=1), plt.Line2D([], [], color=EYE_COLORS[1], lw=1),
               plt.Line2D([], [], color=INK, lw=1.6), plt.Rectangle((0, 0), 1, 1, color=SHADE)]
    fig.legend(handles, ["eye 0", "eye 1", "mean of eyes", "fixation slide"], frameon=False, fontsize=9,
               loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_trial_means(summary: pd.DataFrame) -> Figure:
    """Mean change from baseline over each chart (bar = mean of eyes, dots = each eye)."""
    s = summary.iloc[::-1]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(s) + 1.2))
    ax.barh(y, s.mean_change_mm, height=0.6, color=MUTED, linewidth=0)
    for eye in (0, 1):
        ax.scatter(s[f"mean_change_eye{eye}_mm"], y, s=28, color=EYE_COLORS[eye], zorder=3, label=f"eye {eye}")
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in s.trial], fontsize=9, color=INK)
    ax.set_xlabel("mean change from fixation baseline over the chart (mm)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


MARK_COLOR = "#52514e"


def plot_mark_timelines(trials: pd.DataFrame, marks: pd.DataFrame, fixations: pd.DataFrame,
                        saccades: pd.DataFrame) -> Figure:
    """One row per chart, time from chart onset: fixations (bars), saccades (ticks), marks (spans)."""
    graphs = trials[trials.kind == "graph"].reset_index(drop=True)
    xmax = (graphs.end - graphs.start).max()
    fig, axes = plt.subplots(len(graphs), 1, figsize=(15, 0.78 * len(graphs) + 1.2), sharex=True)
    for ax, g in zip(np.atleast_1d(axes), graphs.itertuples()):
        fx = fixations[(fixations.end > g.start) & (fixations.start < g.end)]
        ax.broken_barh(list(zip(fx.start - g.start, fx.end - fx.start)), (0.55, 0.35), color=FIXATION_COLOR,
                       linewidth=0)
        sc = saccades[(saccades.onset >= g.start) & (saccades.onset < g.end)]
        ax.vlines(sc.onset - g.start, 0.1, 0.45, color=GAP_COLOR, linewidth=1.2)
        for m in marks[marks.chart == g.chart].itertuples():
            ax.axvspan(m.start - g.start, m.end - g.start, color=MARK_COLOR, alpha=0.28 if m.kept else 0.12,
                       linewidth=0)
        ax.axvline(g.end - g.start, color=INK, linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(g.chart, rotation=0, ha="right", va="center", fontsize=9, color=INK)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlim(0, xmax + 0.5)
    ax.set_xlabel("s from chart onset (black line = chart end)", color=MUTED, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FIXATION_COLOR), plt.Line2D([], [], color=GAP_COLOR, lw=1.2),
               plt.Rectangle((0, 0), 1, 1, color=MARK_COLOR, alpha=0.28)]
    fig.legend(handles, ["fixation", "saccade onset", "mark being drawn"], frameon=False, fontsize=9,
               loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_peri_mark(curves: dict[str, dict[str, pd.DataFrame]], baselines: dict[str, float], n_marks: int) -> Figure:
    """Event rates around mark start and end. curves[alignment][event] -> peri_event_rates output."""
    colors = {"saccade": GAP_COLOR, "fixation": FIXATION_COLOR}
    fig, axes = plt.subplots(1, len(curves), figsize=(11, 3.6), sharey=True, squeeze=False)
    for ax, (align, by_event) in zip(axes[0], curves.items()):
        for event, c in by_event.items():
            mid = c.t + (c.t.iloc[1] - c.t.iloc[0]) / 2
            ax.fill_between(mid, c.rate_hz - c.sem_hz, c.rate_hz + c.sem_hz, color=colors[event], alpha=0.15,
                            linewidth=0)
            ax.plot(mid, c.rate_hz, color=colors[event], linewidth=2, label=f"{event} onsets")
            ax.axhline(baselines[event], color=colors[event], linewidth=1, linestyle=(0, (3, 2)))
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_title(f"aligned to mark {align}", loc="left", fontsize=11, color=INK)
        ax.set_xlabel(f"s from mark {align}", color=MUTED, fontsize=9)
        _style(ax)
    axes[0, 0].set_ylabel("events per second", color=INK, fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"n = {n_marks} marks; band = ±1 SEM; dashed = rate over all chart time", x=0.01, ha="left",
                 fontsize=9, color=MUTED)
    fig.tight_layout()
    return fig
