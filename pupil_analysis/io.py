"""Load raw Pupil Core recording data (.pldata + _timestamps.npy)."""

from pathlib import Path

import msgpack
import numpy as np
import pandas as pd


def load_pldata(rec_dir: str | Path, topic: str) -> list[dict]:
    """Return every record in `<rec_dir>/<topic>.pldata` as a dict."""
    rows = []
    with open(Path(rec_dir) / f"{topic}.pldata", "rb") as f:
        for _topic, payload in msgpack.Unpacker(f, raw=False, use_list=False):
            rows.append(msgpack.unpackb(payload, raw=False, use_list=False))
    return rows


def load_world_timestamps(rec_dir: str | Path) -> np.ndarray:
    """One Pupil timestamp per world video frame."""
    return np.load(Path(rec_dir) / "world_timestamps.npy")


def load_frame_lookup(rec_dir: str | Path) -> pd.DataFrame:
    """Pupil Player's frame table: one row per Player frame number.

    Player fills gaps in the world video with placeholder frames, so its
    frame numbers drift ahead of the video file's. Columns:
    `timestamp`, `video_frame` (index in world.mp4, -1 for placeholders).
    Without a `world_lookup.npy` (recording never opened in Player), the
    two numberings are the same.
    """
    path = Path(rec_dir) / "world_lookup.npy"
    if not path.exists():
        ts = load_world_timestamps(rec_dir)
        return pd.DataFrame({"timestamp": ts, "video_frame": np.arange(len(ts))})
    lookup = np.load(path)
    video_frame = np.where(lookup["container_idx"] == -1, -1, lookup["container_frame_idx"])
    return pd.DataFrame({"timestamp": lookup["timestamp"], "video_frame": video_frame})


def timestamps_to_frames(timestamps, world_ts: np.ndarray) -> np.ndarray:
    """Index of the world frame each timestamp falls into (-1 if before the first frame)."""
    return np.searchsorted(world_ts, np.asarray(timestamps), side="right") - 1


def player_gaze_path(rec_dir: str | Path) -> Path | None:
    """Newest post-hoc gaze mapping computed by Pupil Player, if any."""
    mappings = sorted(
        (Path(rec_dir) / "offline_data" / "gaze-mappings").glob("*.pldata"), key=lambda p: p.stat().st_mtime
    )
    return mappings[-1] if mappings else None


def load_gaze(rec_dir: str | Path, min_confidence: float = 0.6, source: str = "recorded") -> pd.DataFrame:
    """Gaze samples with normalized position and the Pupil Player frame they belong to.

    source="recorded" uses gaze.pldata from Pupil Capture; source="player"
    uses Pupil Player's post-hoc gaze mapping (what Player draws when
    post-hoc calibration is on).
    `x`, `y` are Pupil's norm_pos: origin bottom-left, range 0-1.
    """
    if source == "recorded":
        gaze = load_pldata(rec_dir, "gaze")
    elif source == "player":
        path = player_gaze_path(rec_dir)
        if path is None:
            raise FileNotFoundError(f"no Pupil Player gaze mapping in {rec_dir}/offline_data/gaze-mappings")
        gaze = load_pldata(path.parent, path.stem)
    else:
        raise ValueError(f"unknown gaze source {source!r}")
    df = pd.DataFrame(
        {
            "timestamp": [g["timestamp"] for g in gaze],
            "x": [g["norm_pos"][0] for g in gaze],
            "y": [g["norm_pos"][1] for g in gaze],
            "confidence": [g["confidence"] for g in gaze],
        }
    )
    df = df[df.confidence >= min_confidence].sort_values("timestamp")
    df["frame"] = timestamps_to_frames(df.timestamp, load_frame_lookup(rec_dir).timestamp.to_numpy())
    return df[df.frame >= 0].reset_index(drop=True)
