import cv2
import numpy as np
import pandas as pd

from pupil_analysis.video import Clip, render_gaze_clips


def write_world_video(rec_dir, n_frames=20, size=(64, 48)):
    """Synthetic world video whose frame i has brightness i * 10."""
    out = cv2.VideoWriter(str(rec_dir / "world.mp4"), cv2.VideoWriter.fourcc(*"avc1"), 30, size)
    for i in range(n_frames):
        out.write(np.full((size[1], size[0], 3), i * 10, np.uint8))
    out.release()
    np.save(rec_dir / "world_timestamps.npy", np.arange(n_frames) / 30)


def write_lookup_with_gap(rec_dir, n_frames=20, gap_after=5, n_fake=3):
    """world_lookup.npy as Pupil Player writes it, with `n_fake` placeholder frames after video frame `gap_after`."""
    dtype = [("container_idx", "<i8"), ("container_frame_idx", "<i8"), ("timestamp", "<f8"), ("pts", "<i8")]
    rows = [(0, i, i / 30, 0) for i in range(gap_after + 1)]
    rows += [(-1, -1, (gap_after + 0.5) / 30 + k * 1e-4, 0) for k in range(n_fake)]
    rows += [(0, i, i / 30, 0) for i in range(gap_after + 1, n_frames)]
    np.save(rec_dir / "world_lookup.npy", np.array(rows, dtype=dtype))


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            return frames
        frames.append(frame)


def corner_brightness(frame):
    return int(frame[-1, -1].mean())


def test_clips_have_requested_frame_counts(tmp_path):
    write_world_video(tmp_path)
    gaze = pd.DataFrame({"x": [0.5], "y": [0.5], "frame": [3]})
    written = render_gaze_clips(tmp_path, [Clip("a", 2, 5), Clip("b", 4, 9)], tmp_path / "out", gaze=gaze)
    assert written == {"a": 4, "b": 6}
    for name, n in written.items():
        assert len(read_frames(tmp_path / "out" / f"{name}.mp4")) == n
    # first frame of clip "b" is source frame 4
    assert abs(corner_brightness(read_frames(tmp_path / "out" / "b.mp4")[0]) - 40) < 8


def test_clip_ranges_use_player_frame_numbers(tmp_path):
    # Player frames 0-5 = video 0-5, 6-8 = placeholders, 9 = video 6, ...
    write_world_video(tmp_path)
    write_lookup_with_gap(tmp_path)
    gaze = pd.DataFrame({"x": [0.5], "y": [0.5], "frame": [9]})
    written = render_gaze_clips(tmp_path, [Clip("gap", 5, 10), Clip("after", 12, 14)], tmp_path / "out", gaze=gaze)
    assert written == {"gap": 6, "after": 3}
    gap = [corner_brightness(f) for f in read_frames(tmp_path / "out" / "gap.mp4")]
    # video frame 5, three gray placeholders, video frames 6 and 7
    for got, want in zip(gap, [50, 128, 128, 128, 60, 70]):
        assert abs(got - want) < 8
    # Player frame 12 is video frame 9
    assert abs(corner_brightness(read_frames(tmp_path / "out" / "after.mp4")[0]) - 90) < 8
