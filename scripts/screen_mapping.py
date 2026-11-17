"""Map gaze onto the tablet screen recording via the AprilTags on the tablet bezel, and check the mapping.

    uv run python scripts/screen_mapping.py recordings/p1

Steps (see `pupil_analysis.surface`):
1. Tags 4-7: Pupil Player's detections (`exports/*/surfaces/marker_detections.csv`), filled in with OpenCV's
   detector where Player missed one (cached in `<pid>_tags_opencv.csv`). Corners are undistorted with
   `world.intrinsics`.
2. A fixed model of the 16 tag corners in a common plane, from frames that see all four tags.
3. The display: during Pupil Capture's validation markers the whole screen is white, so the display is the
   bright quadrilateral in the world image. Its corners are measured in every such frame, mapped into
   the tag plane and median-combined, then sent to the screen recording's content corners
   (2386 x 1668 px above the letterbox). This makes the tag model a table of screen-recording px.
4. Per world frame (>= 2 tags): the camera pose relative to the tablet (PnP with the calibrated camera;
   screen px are a metric frame of the plane), giving a homography from undistorted world px to screen px.
   `--method homography` fits a free 8-DOF homography instead; it extrapolates badly when only the two
   right-hand tags are visible (p1: up to 205 px at the far marker).
5. Gaze: Pupil Capture's recorded gaze, offset-corrected at the validations (`pupil_analysis.gaze`), mapped
   per sample.

Checks:
- Geometry only: Pupil Capture logs where each validation marker was detected in the world image
  (`validation.data` ref_list). Mapped to the screen, it should land on the marker's centre in the screen
  recording. Gaze is not involved.
- End to end: mapped gaze during each fixation cross vs the cross's position in the screen recording.

Writes `<pid>_gaze_screen.csv`, `<pid>_screen_mapping_check.csv`, `<pid>_screen_mapping.json`
(model, display corners, check summary, provenance) and `<pid>_screen_mapping_check.png`.
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pupil_analysis.gaze import correct_gaze, validation_offsets
from pupil_analysis.io import load_frame_lookup, load_gaze, load_pldata, timestamps_to_frames
from pupil_analysis.provenance import write_record
from pupil_analysis.surface import (
    apply_h,
    detect_tags_opencv,
    display_quad,
    fit_tag_model,
    frame_homographies,
    frame_poses,
    load_intrinsics,
    load_player_tags,
    map_points,
    merge_tags,
    model_to_screen,
    undistorted_corners,
)

SCREEN_WH = (2386, 1668)  # screen-recording content above the letterbox (measured: first black row 1668)
SCREEN_FULL_H = 1964
MIN_TAGS = 2
DISPLAY_MIN_TAGS = 3  # frames used to locate the display corners


def screen_gray(video: Path, start_s: float, dur_s: float, fps: float, scale: int = 2) -> np.ndarray:
    """Grey frames of the screen-recording content at `fps` from `start_s`, downscaled by `scale`."""
    w, h = SCREEN_WH[0] // scale, SCREEN_WH[1] // scale
    vf = f"fps={fps},crop={SCREEN_WH[0]}:{SCREEN_WH[1]}:0:0,scale={w}:{h}"
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start_s:.4f}", "-t", f"{dur_s:.4f}", "-i", str(video), "-an",
           "-vf", vf, "-pix_fmt", "gray", "-f", "rawvideo", "-"]  # fmt: skip
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w)


def dark_blob_centre(g: np.ndarray, thresh: int = 60, min_area: int = 50, top: int = 0) -> tuple[float, float] | None:
    """Centroid of the largest dark connected component below row `top` (in g's px)."""
    bw = (g < thresh).astype(np.uint8)
    bw[:top] = 0
    n, _, stats, cent = cv2.connectedComponentsWithStats(bw)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (float(cent[k][0]), float(cent[k][1])) if stats[k, cv2.CC_STAT_AREA] >= min_area else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("participant_dir", type=Path)
    ap.add_argument("--method", choices=["pose", "homography"], default="pose",
                    help="per-frame mapping: camera pose (6 DOF, default) or free homography (8 DOF)")
    ap.add_argument("--redetect", action="store_true", help="rerun OpenCV tag detection")
    ap.add_argument("--min-confidence", type=float, default=0.6,
                    help="gaze confidence floor for the mapped samples (default 0.6); validation offsets use 0.8")
    args = ap.parse_args()
    pdir: Path = args.participant_dir
    pid = pdir.name
    rec = next(p.parent for p in pdir.glob("*/info.player.json"))
    timeline = pd.read_csv(pdir / f"{pid}_event_timeline.csv")
    sync = pd.read_csv(pdir / f"{pid}_sync_check.csv").groupby("clock").fitted_offset_s.first()
    screen_off, world_lag = sync["screen_recording"], sync["world_video"]
    K, D = load_intrinsics(rec)

    # 1. tags
    cache = pdir / f"{pid}_tags_opencv.csv"
    if args.redetect or not cache.exists():
        print("detecting tags with OpenCV ...")
        detect_tags_opencv(rec).to_csv(cache, index=False)
    tags = merge_tags(load_player_tags(rec), pd.read_csv(cache))
    tags_ud = undistorted_corners(tags, K, D)
    n_per = tags.groupby("world_index").size()
    print(f"tags: {len(tags)} detections in {len(n_per)} frames "
          f"({(tags.source == 'opencv').sum()} filled in by OpenCV)")  # fmt: skip

    # 2. tag model (plane coordinates)
    model = fit_tag_model(tags_ud)
    homs_plane = frame_homographies(tags_ud, model, MIN_TAGS).set_index("world_index")

    # 3. display corners during the validation markers (screen all white)
    lookup = load_frame_lookup(rec)
    vm = timeline[(timeline.event == "validation_markers") & timeline.in_recording]
    cand = [f for a, b in zip(vm.start_world_frame, vm.end_world_frame) for f in range(int(a), int(b) + 1)
            if f in homs_plane.index and homs_plane.at[f, "n_tags"] >= DISPLAY_MIN_TAGS]  # fmt: skip
    video_frame = lookup.video_frame.to_numpy()
    want = {int(video_frame[f]): f for f in cand if video_frame[f] >= 0}
    quads = []
    cap = cv2.VideoCapture(str(rec / "world.mp4"))
    try:
        for v in range(max(want) + 1 if want else 0):
            if not cap.grab():
                break
            if v in want:
                q = display_quad(cv2.cvtColor(cap.retrieve()[1], cv2.COLOR_BGR2GRAY), K, D)
                if q is not None:
                    quads.append(apply_h(homs_plane.at[want[v], "H"], q))
    finally:
        cap.release()
    if not quads:
        raise SystemExit("display corners not found in any validation frame")
    quads = np.array(quads)
    display_model = np.median(quads, axis=0)
    model_screen, G = model_to_screen(model, display_model, SCREEN_WH)
    per_frame = np.array([apply_h(G, q) for q in quads])
    corner_spread = np.median(np.linalg.norm(per_frame - apply_h(G, display_model), axis=2), axis=0)
    print(f"display corners from {len(quads)}/{len(cand)} frames; median per-frame deviation (screen px) "
          f"TL/TR/BR/BL: {np.round(corner_spread, 1).tolist()}")  # fmt: skip

    # 4. per-frame mapping to screen px: camera pose from the tags (PnP), or a free homography
    if args.method == "pose":
        homs = frame_poses(tags_ud, model_screen, K, MIN_TAGS)
    else:
        homs = frame_homographies(tags_ud, model_screen, MIN_TAGS)

    # 5. gaze
    gaze = load_gaze(rec, min_confidence=args.min_confidence, source="recorded")
    offsets = validation_offsets(rec, load_gaze(rec, min_confidence=0.8, source="recorded"))
    gaze_c = correct_gaze(gaze, offsets)
    gs = map_points(gaze_c, homs, K, D)
    gs = gs.rename(columns={"x": "x_norm", "y": "y_norm"})
    out_cols = ["timestamp", "frame", "confidence", "x_norm", "y_norm", "sx", "sy", "n_tags", "tags"]
    gs[out_cols].to_csv(pdir / f"{pid}_gaze_screen.csv", index=False)
    print(f"gaze: {gs.sx.notna().mean():.1%} of {len(gs)} samples mapped")

    # check A: validation markers (geometry only)
    checks = []
    video = pdir / f"{pid}_screen.mov"
    notes = [n for n in load_pldata(rec, "notify") if n.get("subject") == "validation.data"]
    for k, note in enumerate(notes):
        ref = pd.DataFrame({"timestamp": [r["timestamp"] for r in note["ref_list"]],
                            "x": [r["norm_pos"][0] for r in note["ref_list"]],
                            "y": [r["norm_pos"][1] for r in note["ref_list"]]})  # fmt: skip
        ref["frame"] = timestamps_to_frames(ref.timestamp, lookup.timestamp.to_numpy())
        ref = map_points(ref, homs, K, D)
        t0 = ref.timestamp.min() + screen_off - world_lag  # screen frame the world camera is showing
        dur = ref.timestamp.max() - ref.timestamp.min() + 0.1
        if t0 < 0:
            continue
        fr = screen_gray(video, t0, dur, 30)
        idx = np.clip(np.round((ref.timestamp - ref.timestamp.min()) * 30).astype(int), 0, len(fr) - 1)
        cent = {i: dark_blob_centre(fr[i]) or (np.nan, np.nan) for i in np.unique(idx)}
        ref["mx"] = [cent[i][0] * 2 for i in idx]
        ref["my"] = [cent[i][1] * 2 for i in idx]
        # one row per marker position (marker jumps > 100 px)
        pos_id = (np.hypot(ref.mx.diff(), ref.my.diff()) > 100).cumsum()
        for _, g in ref.groupby(pos_id):
            g = g.dropna(subset=["sx", "mx"])
            if len(g) < 5:
                continue
            checks.append({"check": "validation_marker", "validation": k, "pupil_s": g.timestamp.median(),
                           "n": len(g), "tags": g.tags.mode().iloc[0],
                           "target_x": g.mx.median(), "target_y": g.my.median(),
                           "mapped_x": g.sx.median(), "mapped_y": g.sy.median()})  # fmt: skip

    # check B: fixation crosses (gaze + geometry)
    crosses = timeline[(timeline.event == "fixation_cross") & timeline.in_recording]
    for c in crosses.itertuples():
        t0 = c.start_pupil_s + screen_off + 1.0
        g0 = screen_gray(video, t0, 0.1, 10)[0]
        cent = dark_blob_centre(g0, thresh=40, top=150)  # skip the menu/tool bars
        sel = gs[(gs.timestamp >= c.start_pupil_s + 1.0) & (gs.timestamp < c.end_pupil_s) & gs.sx.notna()]
        if cent is None or len(sel) < 20:
            continue
        checks.append({"check": "fixation_cross", "validation": np.nan, "pupil_s": c.start_pupil_s,
                       "n": len(sel), "tags": sel.tags.mode().iloc[0], "block": c.block, "task": c.task,
                       "target_x": cent[0] * 2, "target_y": cent[1] * 2,
                       "mapped_x": sel.sx.median(), "mapped_y": sel.sy.median()})  # fmt: skip
    ck = pd.DataFrame(checks)
    ck["err_x"], ck["err_y"] = ck.mapped_x - ck.target_x, ck.mapped_y - ck.target_y
    ck["err_px"] = np.hypot(ck.err_x, ck.err_y)
    ck.to_csv(pdir / f"{pid}_screen_mapping_check.csv", index=False)
    summary = {}
    for name, g in ck.groupby("check"):
        summary[name] = {"n": len(g), "median_err_px": round(g.err_px.median(), 1),
                         "max_err_px": round(g.err_px.max(), 1),
                         "median_err_x_px": round(g.err_x.median(), 1), "median_err_y_px": round(g.err_y.median(), 1)}  # fmt: skip
        if name == "validation_marker":
            summary[name]["per_validation_median_px"] = g.groupby("validation").err_px.median().round(1).to_dict()
        print(name, summary[name])

    # figure
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.add_patch(plt.Rectangle((0, 0), *SCREEN_WH, fill=False, color="k"))
    for name, mk in (("validation_marker", "o"), ("fixation_cross", "+")):
        g = ck[ck.check == name]
        ax.scatter(g.target_x, g.target_y, marker=mk, color="k", s=60, label=f"{name} target")
        ax.scatter(g.mapped_x, g.mapped_y, marker=mk, color="C3", s=40, label=f"{name} mapped")
        for r in g.itertuples():
            ax.plot([r.target_x, r.mapped_x], [r.target_y, r.mapped_y], color="C3", lw=0.8)
    for t, c in model_screen.items():
        ax.fill(*c.T, color="C0", alpha=0.6)
        mx, my = c.mean(0)
        ax.text(float(mx), float(my), str(t), ha="center", va="center", color="w")
    ax.set_xlim(-200, SCREEN_WH[0] + 200)
    ax.set_ylim(SCREEN_WH[1] + 200, -200)
    ax.set_aspect("equal")
    ax.set_title(f"{pid}: world -> screen-recording mapping check (px)")
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(pdir / f"{pid}_screen_mapping_check.png", dpi=120)

    rec_path = write_record(
        pdir / f"{pid}_screen_mapping.json",
        {
            "inputs": {"recording": rec, "screen_recording": video, "tag_cache": cache},
            "screen_content_px": {"width": SCREEN_WH[0], "height": SCREEN_WH[1], "full_height": SCREEN_FULL_H,
                                  "how": "measured: first all-black row of the screen recording"},
            "tags": {"detections": len(tags), "frames": len(n_per),
                     "filled_by_opencv": int((tags.source == "opencv").sum()),
                     "frames_with_homography": len(homs), "min_tags": MIN_TAGS, "method": args.method,
                     "model_screen_px": {int(t): np.round(c, 2).tolist() for t, c in model_screen.items()}},
            "display": {"how": "automatic: bright quadrilateral in world frames during the validation markers, "
                               "edges fit after undistortion, corners mapped to the tag plane and median-combined",
                        "frames_used": len(quads), "frames_tried": len(cand), "min_tags": DISPLAY_MIN_TAGS,
                        "median_per_frame_deviation_px_TL_TR_BR_BL": np.round(corner_spread, 1).tolist()},
            "gaze": {"source": "recorded", "corrected": "validation offsets, interpolated in time",
                     "min_confidence": args.min_confidence, "min_confidence_for_offsets": 0.8,
                     "mapped_fraction": round(float(gs.sx.notna().mean()), 4),
                     "timing": "each gaze sample uses the homography of the world frame it falls in"},
            "checks": summary,
            "world_lag_s": world_lag, "screen_offset_s": screen_off,
        },
    )  # fmt: skip
    print(f"wrote {rec_path}")


if __name__ == "__main__":
    main()
