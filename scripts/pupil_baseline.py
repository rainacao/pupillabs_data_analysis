"""Pupil size per Mini-VLAT chart in recording 003, baseline-corrected against the preceding fixation slide.

uv run python scripts/pupil_baseline.py             # plots + CSVs into recordings/003/pupil/
uv run python scripts/pupil_baseline.py --detect    # re-detect fixation slides from the world video and print them

Outputs: pupil_timeline.png, pupil_trial_traces.png, pupil_trial_means.png,
pupil_trial_summary.csv, pupil_trial_traces.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from pupil_analysis.io import load_frame_lookup
from pupil_analysis.plots import plot_pupil_timeline, plot_trial_means, plot_trial_traces
from pupil_analysis.pupil import Trial, load_pupil_diameter, low_segments, trial_summary, trial_traces

REC = Path("recordings/003")
BASELINE_S = 1.0  # baseline = mean diameter over the last second of the fixation slide

# Pupil Player frames: (fixation slide start, chart onset, chart end).
# Fixation slides 2-12 come from `--detect` (grey slide = no white chart panel in the
# world video) and agree with the Player annotations within 1-10 frames where those exist.
# Slide 1 (1822-1965) has a settings panel open beside it, so it was set by eye.
TRIALS = [
    Trial("1_treemap", 1822, 1965, 2480),
    Trial("2_100%_stacked_bar", 2480, 2635, 2942),
    Trial("3_histogram", 2942, 3097, 3463),
    Trial("4_choropleth", 3463, 3617, 4030),
    Trial("5_pie", 4030, 4185, 4900),
    Trial("6_bubble", 4900, 5053, 5460),
    Trial("7_stacked_bar", 5460, 5617, 6118),
    Trial("8_line", 6118, 6275, 7102),
    Trial("9_bar", 7102, 7261, 7522),
    Trial("10_area", 7522, 7683, 8211),
    Trial("11_stacked_area", 8211, 8367, 9476),
    Trial("12_scatter", 9476, 9633, 9782),
]


def detect() -> None:
    from pupil_analysis.video import white_fraction

    for start, end in low_segments(white_fraction(REC), threshold=0.016, min_len=60):
        print(f"fixation slide: frames {start}-{end}")
    annotations = REC / "exports" / "000" / "annotations.csv"
    if annotations.exists():
        print("Player annotations at frames:", pd.read_csv(annotations)["index"].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect", action="store_true")
    args = parser.parse_args()
    if args.detect:
        detect()
        return

    out_dir = REC / "pupil"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_ts = load_frame_lookup(REC).timestamp.to_numpy()
    pupil = load_pupil_diameter(REC)

    traces = trial_traces(pupil, TRIALS, frame_ts, BASELINE_S)
    summary = trial_summary(pupil, TRIALS, frame_ts, BASELINE_S)
    traces.to_csv(out_dir / "pupil_trial_traces.csv", index=False)
    summary.to_csv(out_dir / "pupil_trial_summary.csv", index=False)
    plot_pupil_timeline(pupil, TRIALS, frame_ts).savefig(out_dir / "pupil_timeline.png", dpi=150)
    plot_trial_traces(traces).savefig(out_dir / "pupil_trial_traces.png", dpi=150)
    plot_trial_means(summary).savefig(out_dir / "pupil_trial_means.png", dpi=150)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(summary.to_string(index=False))
    print(f"wrote {out_dir}/")


if __name__ == "__main__":
    main()
