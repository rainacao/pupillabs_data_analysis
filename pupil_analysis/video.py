"""Cut world-video clips with gaze drawn on top.

Frame numbers are Pupil Player's (see `load_frame_lookup`), so ranges read
off Player's timeline select the same frames here.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pupil_analysis.io import load_frame_lookup, load_gaze


@dataclass(frozen=True)
class Clip:
    name: str
    start: int  # first Pupil Player frame, inclusive
    end: int  # last Pupil Player frame, inclusive


def draw_gaze(frame, xs: np.ndarray, ys: np.ndarray) -> None:
    """Draw raw gaze samples (green dots) and their mean (red ring) in place."""
    h, w = frame.shape[:2]
    for x, y in zip(xs, ys):
        cv2.circle(frame, (int(x * w), int((1 - y) * h)), 4, (0, 255, 0), -1)
    mx, my = xs.mean(), ys.mean()
    cv2.circle(frame, (int(mx * w), int((1 - my) * h)), 20, (0, 0, 255), 3)


def render_gaze_clips(
    rec_dir: str | Path,
    clips: list[Clip],
    out_dir: str | Path,
    gaze: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Write one `<out_dir>/<clip.name>.mp4` per clip, with gaze overlaid.

    Reads the world video sequentially in a single pass: OpenCV's frame
    seeking is inaccurate on Pupil's variable-frame-rate videos. Player's
    placeholder frames (gaps in the world video) are written as gray
    frames, as Player shows them. Returns the number of frames written per clip.
    """
    rec_dir, out_dir = Path(rec_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if gaze is None:
        gaze = load_gaze(rec_dir)
    # Player frame -> (xs, ys) of the gaze samples in that frame
    by_frame = {f: (grp.x.to_numpy(), grp.y.to_numpy()) for f, grp in gaze.groupby("frame")}
    video_frame = load_frame_lookup(rec_dir).video_frame.to_numpy()

    cap = cv2.VideoCapture(str(rec_dir / "world.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc(*"avc1")  # H.264, plays in QuickTime
    writers = {c.name: cv2.VideoWriter(str(out_dir / f"{c.name}.mp4"), fourcc, fps, (w, h)) for c in clips}
    written = {c.name: 0 for c in clips}
    last_frame = min(max(c.end for c in clips), len(video_frame) - 1)

    try:
        next_video_frame = 0
        for i in range(last_frame + 1):
            if video_frame[i] == -1:
                frame = np.full((h, w, 3), 128, np.uint8)
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                assert next_video_frame == video_frame[i], "world.mp4 out of step with world_lookup.npy"
                next_video_frame += 1
            active = [c for c in clips if c.start <= i <= c.end]
            if not active:
                continue
            if i in by_frame:
                draw_gaze(frame, *by_frame[i])
            cv2.putText(frame, f"frame {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            for c in active:
                writers[c.name].write(frame)
                written[c.name] += 1
    finally:
        cap.release()
        for wr in writers.values():
            wr.release()
    return written


def render_gaze_clip(rec_dir: str | Path, start: int, end: int, out_path: str | Path) -> int:
    """Write Player frames `start`..`end` (inclusive) of the world video with gaze overlaid."""
    out_path = Path(out_path)
    return render_gaze_clips(rec_dir, [Clip(out_path.stem, start, end)], out_path.parent)[out_path.stem]


def white_fraction(rec_dir: str | Path, step: int = 2, smooth: int = 9) -> np.ndarray:
    """Per Player frame: fraction of near-white, unsaturated pixels in the world video.

    A white chart panel on the tablet raises it; a grey fixation-cross slide
    drops it, which makes it a cheap slide-transition detector. Frames are
    sampled every `step`, mapped onto Player frames (placeholders interpolated),
    and median-smoothed over `smooth` frames.
    """
    rec_dir = Path(rec_dir)
    video_frame = load_frame_lookup(rec_dir).video_frame.to_numpy()
    cap = cv2.VideoCapture(str(rec_dir / "world.mp4"))
    values = np.full(int(video_frame.max()) + 1, np.nan)
    i = 0
    try:
        while cap.grab():
            if i % step == 0:
                _, frame = cap.retrieve()
                hsv = cv2.cvtColor(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
                values[i] = np.mean((hsv[..., 2] > 225) & (hsv[..., 1] < 25))
            i += 1
    finally:
        cap.release()
    per_player = np.array([values[v] if 0 <= v < len(values) else np.nan for v in video_frame])
    s = pd.Series(per_player).interpolate(limit_area="inside").bfill().ffill()
    return s.rolling(smooth, center=True, min_periods=1).median().to_numpy()
