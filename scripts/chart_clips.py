"""Render gaze clips and count fixations/saccades for the mini-VLAT charts in recording 004.

uv run python scripts/chart_clips.py            # clips + counts
uv run python scripts/chart_clips.py --no-video # counts only
"""

import argparse
from pathlib import Path

from pupil_analysis.events import count_events, load_fixations_csv, load_fixations_pldata
from pupil_analysis.io import load_frame_lookup, load_gaze, player_gaze_path
from pupil_analysis.plots import plot_event_histograms
from pupil_analysis.video import Clip, render_gaze_clips

REC = Path("recordings/004")
# Pupil Player frame numbers (as shown on Player's timeline)
CHARTS = {
    "1_stacked_area": (1531, 2242),
    "2_area": (2243, 2654),
    "3_stacked_bar": (4437, 5169),
}


def load_fixations(rec_dir: Path):
    """Prefer the newest Pupil Player export, then Player's offline detector, then the online fixations."""
    exports = sorted(rec_dir.glob("exports/*/fixations.csv"))
    if exports:
        print(f"fixations from {exports[-1]}")
        return load_fixations_csv(exports[-1])
    if (rec_dir / "offline_data" / "fixations.pldata").exists():
        print("fixations from Pupil Player's offline detector (offline_data/fixations.pldata)")
        return load_fixations_pldata(rec_dir)
    print("no Pupil Player fixations found; using online fixations.pldata (coarse durations)")
    return load_fixations_pldata(rec_dir, offline=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    out_dir = REC / "clips"
    if not args.no_video:
        source = "player" if player_gaze_path(REC) else "recorded"
        print(f"gaze source: {source}")
        clips = [Clip(name, start, end) for name, (start, end) in CHARTS.items()]
        for name, n in render_gaze_clips(REC, clips, out_dir, gaze=load_gaze(REC, source=source)).items():
            print(f"{out_dir / name}.mp4: {n} frames")

    frame_ts = load_frame_lookup(REC).timestamp.to_numpy()
    fixations = load_fixations(REC)
    counts = count_events(fixations, CHARTS, frame_ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_dir / "event_counts.csv", index=False)
    plot_event_histograms(fixations, CHARTS, frame_ts).savefig(out_dir / "event_histograms.png", dpi=150)
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
