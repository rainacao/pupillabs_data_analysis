"""Side-by-side video for one task: world camera + gaze, tablet screen recording, and the pupil trace.

    uv run python scripts/sync_video.py recordings/p1 area trends
    uv run python scripts/sync_video.py recordings/p1 stackarea value-absolute --pad 2

Writes `<participant>/videos/<pid>_<block>-<task>_sync.mp4` (60 fps, Pupil-time grid), covering
the task's fixation cross through the end of the chart, plus `--pad` seconds each side.

Layout:
- Top left: world camera, with the gaze samples of each frame (green) and their mean (red ring),
  the current fixation (yellow ring, numbered), and a running list of the latest fixations.
  Gaze is Pupil Capture's recorded gaze by default (`--gaze player` for Pupil Player's post-hoc
  mapping). Both are corrected for the offset measured at the Pupil Capture validations,
  interpolated in time (see `pupil_analysis.gaze`); `--no-gaze-correction` turns that off.
  Fixations come from the newest Pupil Player export (`exports/*/fixations.csv`). They were detected
  on Player's gaze, so their positions are corrected with Player-gaze offsets.
- Top right: the tablet screen recording, with gaze mapped onto it through the bezel AprilTags
  (`<pid>_gaze_screen.csv` from `scripts/screen_mapping.py`; drawn for the default recorded, corrected gaze).
  Each output frame shows the gaze samples of the world frame shown on the left.
- Middle strip: time from chart onset, current slide, pen up/down (annotation strokes), and each
  eye's % change from its fixation-cross baseline plus their mean.
- Bottom: the same pupil trace as `<pid>_pupil_dilation.png` (2D diameter, 50 ms bins, mean of
  eyes smoothed 250 ms), with stroke bands and a moving cursor.

Sync: everything is placed on Pupil time.
- World frames, gaze and pupil data are already in Pupil time.
- Stroke and slide times come from `<pid>_event_timeline.csv`.
- The screen recording is placed using the offset in `<pid>_sync_check.csv`, then delayed by the
  measured world-camera lag (`world_video` offset, ~98 ms on p1). This makes both video panels
  change slides on the same frame. The pupil trace and stroke indicator are not shifted, so they
  match the dilation plots and CSV.

Pupil Core's eye0 camera films the right eye and eye1 the left eye.

Confidence floors: `--min-confidence` (default 0.8) for the pupil trace, read together with the
matching `<pid>_pupil_dilation_conf<NN>.csv` (run task_pupil_dilation.py with the same value first);
`--gaze-min-confidence` (default 0.6) for the drawn gaze, on both panels. The screen panel can only show
samples `screen_mapping.py` mapped, so run it with a floor at or below this one. Non-default values add
`_conf<NN>` / `_gaze<NN>` to the video name. Validation offsets always use gaze with confidence >= 0.8.

Every shift applied to a video, the parameters behind it, and the code version are written to
`<video>.provenance.json` next to it.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg

from pupil_analysis.gaze import correct_gaze, validation_offsets
from pupil_analysis.io import load_frame_lookup, load_gaze
from pupil_analysis.provenance import write_record
from pupil_analysis.pupil import DEFAULT_MIN_CONFIDENCE, binned_trace, confidence_suffix, load_pupil_diameter
from pupil_analysis.video import draw_gaze

FPS = 60
GAZE_MIN_CONFIDENCE = 0.6  # drawn gaze (load_gaze's default)
BIN_S, PEAK_SMOOTH_S = 0.05, 0.25  # same as task_pupil_dilation.py
TOP_H = 720  # height of the two video panels
STRIP_H = 110
PLOT_H = 340
BLACK_LEVEL = 16  # a screen-recording row is letterbox if no pixel in it exceeds this in any sampled frame
EYE_LABEL = {0: "eye0 (right)", 1: "eye1 (left)"}
COLORS_BGR = {0: (180, 119, 31), 1: (14, 127, 255), "mean": (0, 0, 0)}  # matplotlib C0, C1, black


def screen_content_size(video: Path, samples: int = 5) -> tuple[int, int]:
    """(width, height) of the screen recording above its black bottom letterbox, measured.

    The content ends at the last row that is brighter than `BLACK_LEVEL` in any of `samples` frames spread
    over the recording. iPad (p1): 2386 x 1964 recording, content 2386 x 1668. A Mac screen recording
    (p3, desktop monitor) has no letterbox, so the whole frame is content.
    """
    probe = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
             "-of", "default=nw=1", str(video)]  # fmt: skip
    info = dict(line.split("=") for line in subprocess.run(probe, capture_output=True, text=True, check=True)
                .stdout.split())  # fmt: skip
    w, h, dur = int(info["width"]), int(info["height"]), float(info["duration"])
    row_max = np.zeros(h, np.uint8)
    for k in range(samples):
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{dur * (k + 0.5) / samples:.2f}", "-i", str(video),
               "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-"]  # fmt: skip
        frame = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.uint8).reshape(h, w)
        row_max = np.maximum(row_max, frame.max(axis=1))
    return w, int(np.flatnonzero(row_max > BLACK_LEVEL)[-1]) + 1


def screen_frames(video: Path, start_s: float, n: int, size: tuple[int, int], content: tuple[int, int]) -> np.ndarray:
    """`n` frames of the screen recording at 60 fps from `start_s`, cropped to `content`, resized to `size`."""
    w, h = size
    vf = f"fps={FPS},crop={content[0]}:{content[1]}:0:0,scale={w}:{h}"
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(start_s, 0):.4f}", "-i", str(video), "-an", "-vf", vf]
    cmd += ["-frames:v", str(n), "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    if len(frames) < n:  # past the end of the recording: pad with black
        frames = np.concatenate([frames, np.zeros((n - len(frames), h, w, 3), np.uint8)])
    return frames


def draw_screen_gaze(img, xs: np.ndarray, ys: np.ndarray) -> None:
    """Gaze mapped onto the screen recording (panel px): samples as green dots, their mean as a red ring."""
    for x, y in zip(xs, ys):
        cv2.circle(img, (int(x), int(y)), 3, (0, 200, 0), -1)
    cv2.circle(img, (int(xs.mean()), int(ys.mean())), 14, (0, 0, 255), 2)


def world_frames(rec: Path, frame_idx: np.ndarray) -> dict[int, np.ndarray]:
    """The world-video frames with Player index in `frame_idx`, read sequentially (seeking is unreliable)."""
    video_frame = load_frame_lookup(rec).video_frame.to_numpy()
    wanted = {int(video_frame[i]): int(i) for i in np.unique(frame_idx) if video_frame[i] >= 0}
    out = {}
    cap = cv2.VideoCapture(str(rec / "world.mp4"))
    try:
        for v in range(max(wanted) + 1):
            ok = cap.grab()
            if not ok:
                break
            if v in wanted:
                out[wanted[v]] = cap.retrieve()[1]
    finally:
        cap.release()
    return out


def pupil_series(pupil: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """% change per eye and smoothed mean, binned as in the dilation plot. Index: bin centre, s from chart onset."""
    base = pd.Series({0: row.baseline_eye0_px, 1: row.baseline_eye1_px})
    parts = []
    for t0, t1 in ((row.baseline_start, row.task_start), (row.task_start, row.task_end)):
        tr = binned_trace(pupil, t0, t1, BIN_S, "diameter_px").reindex(columns=[0, 1])
        tr = 100 * tr.sub(base, axis=1).div(base, axis=1)
        tr["mean"] = tr[[0, 1]].mean(axis=1).rolling(round(PEAK_SMOOTH_S / BIN_S), center=True, min_periods=1).mean()
        tr.index = tr.index + BIN_S / 2 + (t0 - row.task_start)
        parts.append(tr)
    return pd.concat(parts)


def plot_panel(series, strokes, row, t_range, width):
    """Static pupil plot as a BGR image, plus a function mapping s-from-onset to pixel x."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, PLOT_H / dpi), dpi=dpi)
    for s in strokes.itertuples():
        ax.axvspan(s.start_pupil_s - row.task_start, s.end_pupil_s - row.task_start, color="C2", alpha=0.3, lw=0)
    ax.plot(series.index, series[0], color="C0", lw=0.8, alpha=0.6, label=EYE_LABEL[0])
    ax.plot(series.index, series[1], color="C1", lw=0.8, alpha=0.6, label=EYE_LABEL[1])
    ax.plot(series.index, series["mean"], color="k", lw=1.6, label="mean of eyes")
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0, color="grey", lw=0.5)
    ax.hlines(row.mean_change_pct, 0, row.task_duration_s, color="C3", label="task mean")
    ax.plot(row.peak_time_s, row.peak_dilation_pct, "v", color="C3", label="peak")
    ax.set_xlim(*t_range)
    ax.set_xlabel("s from chart onset (negative = fixation cross)")
    ax.set_ylabel("% change from baseline")
    ax.legend(loc="lower left", fontsize=8, ncol=5, framealpha=0.8)
    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    y_top, y_bot = (img.shape[0] - ax.transData.transform([(0, y) for y in ax.get_ylim()])[:, 1]).astype(int)[::-1]

    def to_x(t: float) -> int:
        return round(ax.transData.transform((t, 0))[0])

    plt.close(fig)
    return img, to_x, (y_top, y_bot)


def slide_at(t: float, timeline: pd.DataFrame) -> str:
    names = {"fixation_cross": "Fixation cross (baseline)", "graph": "Chart", "validation": "Validation break"}
    ev = timeline[timeline.event.isin(names) & (timeline.start_pupil_s <= t) & (timeline.end_pupil_s > t)]
    if ev.empty:
        return "between slides"
    e = ev.iloc[0]
    label = names[e.event]
    return f"{label}: {e.block} / {e.task}" if e.event == "graph" else label


def put(img, text, org, scale=0.8, color=(0, 0, 0), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def offsets_px(off: pd.DataFrame) -> list[dict]:
    """Validation offsets in world-image pixels (+dy = gaze above the marker)."""
    return [
        {"validation_pupil_s": round(t, 3), "dx_px": round(dx * 1280, 1), "dy_px": round(dy * 720, 1), "n": n}
        for t, dx, dy, n in zip(off.t, off.dx, off.dy, off.n)
    ]


def applied_px(off: pd.DataFrame, times: dict[str, float]) -> dict:
    """The (interpolated) correction subtracted at each named time, in pixels."""
    return {
        name: {"dx_px": round(float(np.interp(t, off.t, off.dx)) * 1280, 1),
               "dy_px": round(float(np.interp(t, off.t, off.dy)) * 720, 1)}
        for name, t in times.items()
    }  # fmt: skip


def load_fixations(rec: Path) -> pd.DataFrame:
    """Pupil Player's exported fixations: `id`, `start`, `end` (Pupil s), `duration_ms`, `x`, `y` (norm_pos)."""
    exports = sorted(rec.glob("exports/*/fixations.csv"))
    if not exports:
        raise FileNotFoundError(f"no fixations.csv in {rec}/exports; export with Player's fixation detector on")
    f = pd.read_csv(exports[-1])
    return pd.DataFrame(
        {
            "id": f.id,
            "start": f.start_timestamp,
            "end": f.start_timestamp + f.duration / 1000,
            "duration_ms": f.duration,
            "timestamp": f.start_timestamp,
            "x": f.norm_pos_x,
            "y": f.norm_pos_y,
        }
    )


def draw_fixation_list(img, fixations: pd.DataFrame, t: float, onset: float, n: int = 8) -> None:
    """Box with the latest `n` fixations started by `t`; the ongoing one in yellow."""
    started = fixations[fixations.start <= t].tail(n)
    x0, y0, w, line_h = 1280 - 330, 50, 318, 26
    h = 34 + line_h * max(len(started), 1)
    roi = img[y0 : y0 + h, x0 : x0 + w]
    roi[:] = (roi * 0.35).astype(np.uint8)  # darken behind the text
    put(img, "fixations (start, duration)", (x0 + 10, y0 + 24), 0.6, (255, 255, 255), 1)
    for k, f in enumerate(started.itertuples()):
        ongoing = f.end > t
        color = (0, 230, 255) if ongoing else (220, 220, 220)
        txt = f"#{f.id:<4} {f.start - onset:+6.2f} s  {f.duration_ms:4.0f} ms" + ("  <" if ongoing else "")
        put(img, txt, (x0 + 10, y0 + 34 + line_h * (k + 1) - 6), 0.6, color, 2 if ongoing else 1)


def boxed_label(img, text, org):
    """White text on a dark box, readable over any video content."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x, y = org
    cv2.rectangle(img, (x - 6, y - h - 8), (x + w + 6, y + base + 4), (40, 40, 40), -1)
    put(img, text, org, 0.7, (255, 255, 255))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("participant_dir", type=Path)
    ap.add_argument("block", help="chart type, e.g. area")
    ap.add_argument("task", help="task name as in the study, e.g. trends or value-absolute")
    ap.add_argument("--pad", type=float, default=1.0, help="seconds before the cross and after the chart")
    ap.add_argument("--gaze", choices=["recorded", "player"], default="recorded", help="gaze source to draw")
    ap.add_argument("--no-gaze-correction", action="store_true", help="draw gaze without the validation offset")
    ap.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                    help=f"pupil confidence floor for the trace (default {DEFAULT_MIN_CONFIDENCE})")
    ap.add_argument("--gaze-min-confidence", type=float, default=GAZE_MIN_CONFIDENCE,
                    help=f"confidence floor for drawn gaze (default {GAZE_MIN_CONFIDENCE})")
    args = ap.parse_args()
    gaze_conf: float = args.gaze_min_confidence
    dil_suffix = confidence_suffix(args.min_confidence)
    video_suffix = dil_suffix + confidence_suffix(gaze_conf, "gaze", GAZE_MIN_CONFIDENCE)
    pdir: Path = args.participant_dir
    pid = pdir.name
    rec = next(p.parent for p in pdir.glob("*/info.player.json"))
    timeline = pd.read_csv(pdir / f"{pid}_event_timeline.csv")
    dilation = pd.read_csv(pdir / f"{pid}_pupil_dilation{dil_suffix}.csv")
    sync = pd.read_csv(pdir / f"{pid}_sync_check.csv").groupby("clock").fitted_offset_s.first()
    screen_off, world_lag = sync["screen_recording"], sync["world_video"]

    match = dilation[(dilation.block == args.block) & (dilation.source_task == args.task)]
    if match.empty:
        raise SystemExit(f"{args.block}/{args.task} not in {pid}_pupil_dilation{dil_suffix}.csv")
    row = match.iloc[0]
    strokes = timeline[
        (timeline.event == "annotation_stroke") & (timeline.block == args.block) & (timeline.task == args.task)
    ]

    t_start, t_end = row.baseline_start - args.pad, row.task_end + args.pad
    T = np.arange(t_start, t_end, 1 / FPS)  # output frame times, Pupil s
    rel = T - row.task_start

    world_ts = load_frame_lookup(rec).timestamp.to_numpy()
    w_idx = np.clip(np.searchsorted(world_ts, T, side="right") - 1, 0, len(world_ts) - 1)
    gaze = load_gaze(rec, min_confidence=gaze_conf, source=args.gaze)
    fixations = load_fixations(rec)
    offsets: dict[str, pd.DataFrame] = {}
    if not args.no_gaze_correction:
        offsets = {name: validation_offsets(rec, load_gaze(rec, min_confidence=0.8, source=name))
                   for name in {args.gaze, "player"}}  # fmt: skip
        for name, off in offsets.items():
            px = [(round(dx * 1280), round(dy * 720)) for dx, dy in zip(off.dx, off.dy)]
            print(f"{name} gaze offset at validations (dx, dy px; +y = up): {px}")
        gaze = correct_gaze(gaze, offsets[args.gaze])
        fixations = correct_gaze(fixations, offsets["player"])  # detected on Player's gaze
    gaze_note = f"{args.gaze} gaze" + ("" if args.no_gaze_correction else ", offset-corrected")
    by_frame = {f: (g.x.to_numpy(), g.y.to_numpy()) for f, g in gaze[gaze.frame.isin(w_idx)].groupby("frame")}
    print(f"reading {len(np.unique(w_idx))} world frames ...")
    world = world_frames(rec, w_idx)

    screen_video = pdir / f"{pid}_screen.mov"
    content_wh = screen_content_size(screen_video)
    print(f"screen recording content: {content_wh[0]} x {content_wh[1]} px")
    screen_w = round(content_wh[0] / content_wh[1] * TOP_H / 2) * 2
    screen_scale = TOP_H / content_wh[1]
    # gaze on the screen panel: mapped by screen_mapping.py (recorded gaze, offset-corrected)
    screen_gaze_path = pdir / f"{pid}_gaze_screen.csv"
    screen_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if args.gaze == "recorded" and not args.no_gaze_correction and screen_gaze_path.exists():
        sg = pd.read_csv(screen_gaze_path)
        if sg.confidence.min() > gaze_conf + 0.01:
            print(f"warning: {screen_gaze_path.name} only has gaze with confidence >= {sg.confidence.min():.2f}; "
                  f"rerun screen_mapping.py --min-confidence {gaze_conf}")
        sg = sg[sg.confidence >= gaze_conf].dropna(subset=["sx"])
        sg = sg[sg.frame.isin(w_idx)]
        screen_by_frame = {int(f): (g.sx.to_numpy() * screen_scale, g.sy.to_numpy() * screen_scale)
                           for f, g in sg.groupby("frame")}  # fmt: skip
    else:
        print(f"no gaze on the screen panel (needs {screen_gaze_path.name} and recorded, corrected gaze)")
    print("reading screen recording ...")
    screen = screen_frames(screen_video, t_start - world_lag + screen_off, len(T), (screen_w, TOP_H), content_wh)

    width = 1280 + screen_w
    pupil = load_pupil_diameter(rec, min_confidence=args.min_confidence, measure="2d")
    series = pupil_series(pupil, row)
    plot_img, to_x, (y_top, y_bot) = plot_panel(series, strokes, row, (rel[0], rel[-1]), width)
    plot_img = cv2.resize(plot_img, (width, PLOT_H))

    out_dir = pdir / "videos"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{pid}_{args.block}-{args.task}_sync{video_suffix}.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"avc1"), FPS, (width, TOP_H + STRIP_H + PLOT_H))
    series_t = series.index.to_numpy()
    try:
        for i, (t, r) in enumerate(zip(T, rel)):
            frame = np.full((TOP_H + STRIP_H + PLOT_H, width, 3), 255, np.uint8)
            wf = world.get(int(w_idx[i]))
            wf = wf.copy() if wf is not None else np.full((720, 1280, 3), 128, np.uint8)
            if int(w_idx[i]) in by_frame:
                draw_gaze(wf, *by_frame[int(w_idx[i])])
            cur = fixations[(fixations.start <= t) & (fixations.end > t)]
            for f in cur.itertuples():
                c = (int(f.x * 1280), int((1 - f.y) * 720))
                cv2.circle(wf, c, 34, (0, 230, 255), 3)
                put(wf, f"#{f.id}", (c[0] + 38, c[1] - 20), 0.7, (0, 230, 255))
            draw_fixation_list(wf, fixations, t, row.task_start)
            frame[:TOP_H, :1280] = wf
            sf = screen[i].copy()
            if int(w_idx[i]) in screen_by_frame:
                draw_screen_gaze(sf, *screen_by_frame[int(w_idx[i])])
            frame[:TOP_H, 1280:] = sf
            boxed_label(frame, f"world camera + {gaze_note}  (frame {w_idx[i]})", (12, 30))
            boxed_label(frame, "screen recording" + (" + mapped gaze" if screen_by_frame else ""),
                        (1292, TOP_H - 16))

            # status strip
            y0 = TOP_H
            put(frame, f"t = {r:+6.2f} s from chart onset", (16, y0 + 40), 0.9)
            put(frame, slide_at(t, timeline), (430, y0 + 40), 0.9)
            active = strokes[(strokes.start_pupil_s <= t) & (strokes.end_pupil_s > t)]
            if len(active):
                cv2.circle(frame, (1150, y0 + 32), 12, (0, 0, 220), -1)
                put(frame, f"PEN DOWN - stroke {int(active.stroke_n.iloc[0])}", (1172, y0 + 42), 0.9, (0, 0, 220))
            else:
                put(frame, f"pen up ({len(strokes)} stroke(s) in this task)", (1150, y0 + 42), 0.7, (120, 120, 120))
            k = np.searchsorted(series_t, r) - 1
            vals = series.iloc[k] if 0 <= k < len(series) and abs(series_t[k] - r) <= BIN_S else None
            x = 16
            for key, label in ((0, EYE_LABEL[0]), (1, EYE_LABEL[1]), ("mean", "mean of eyes")):
                v = vals[key] if vals is not None else np.nan
                txt = f"{label}: {v:+6.1f}%" if np.isfinite(v) else f"{label}:    --  "
                put(frame, txt, (x, y0 + 88), 0.8, COLORS_BGR[key])
                x += 330
            put(frame, "pupil diameter vs fixation-cross baseline (2D, 50 ms bins; mean smoothed 250 ms)",
                (1010, y0 + 88), 0.55, (90, 90, 90), 1)  # fmt: skip

            # plot + cursor
            frame[TOP_H + STRIP_H :] = plot_img
            cx = to_x(r)
            cv2.line(frame, (cx, TOP_H + STRIP_H + y_top), (cx, TOP_H + STRIP_H + y_bot), (0, 0, 220), 2)
            writer.write(frame)
    finally:
        writer.release()
    print(f"wrote {out} ({len(T)} frames, {len(T) / FPS:.1f} s)")

    sync_params_path = pdir / f"{pid}_sync_params.json"
    sync_params = json.loads(sync_params_path.read_text()) if sync_params_path.exists() else None
    moments = {"video_start": t_start, "chart_onset": row.task_start, "video_end": t_end}
    gaze_record: dict[str, object] = {"source": args.gaze, "corrected": not args.no_gaze_correction}
    fix_record: dict[str, object] = {"source": str(max(rec.glob("exports/*/fixations.csv"))),
                  "corrected": not args.no_gaze_correction}  # fmt: skip
    if offsets:
        how = "automatic: median gaze - marker offset at each Pupil Capture validation, linearly interpolated in time"
        gaze_record |= {"how": how, "validation_offsets": offsets_px(offsets[args.gaze]),
                        "subtracted_during_video": applied_px(offsets[args.gaze], moments)}  # fmt: skip
        fix_record |= {"how": how + " (Player-gaze offsets: fixations were detected on Player's gaze)",
                       "validation_offsets": offsets_px(offsets["player"]),
                       "subtracted_during_video": applied_px(offsets["player"], moments)}  # fmt: skip
    record = write_record(
        out.with_suffix(".provenance.json"),
        {
            "video": out,
            "task": {"block": args.block, "task": args.task, "chart_onset_pupil_s": row.task_start,
                     "fixation_cross_pupil_s": [row.baseline_start, row.baseline_end],
                     "chart_end_pupil_s": row.task_end},
            "window": {"start_pupil_s": t_start, "end_pupil_s": t_end, "pad_s": args.pad, "fps": FPS,
                       "frames": len(T), "world_frames": [int(w_idx[0]), int(w_idx[-1])]},
            "shifts_applied": {
                "screen_recording": {
                    "screen_time = pupil_time + offset_s": screen_off,
                    "extra_delay_s": world_lag,
                    "why_delay": "match the world camera, which sees slide changes this much later than reVISit "
                    "logs them; both panels then change slides on the same frame",
                    "screen_time_of_first_frame_s": t_start - world_lag + screen_off,
                    "how": "automatic: both values fitted in event_timeline.py (see sync_params)",
                },
                "gaze": gaze_record,
                "fixations": fix_record,
            },
            "not_shifted": [
                "world frames, gaze and pupil samples: native Pupil timestamps",
                (
                    "pupil trace, stroke bands and PEN DOWN indicator: event-timeline times (reVISit clock "
                    "converted to Pupil time), not delayed by the world-camera lag, so they match the "
                    "dilation plots and CSV"
                ),
            ],
            "parameters": {
                "pupil": {"measure": "2d (ellipse major axis, px)", "cleaning": "kret (clean_eye)",
                          "min_confidence": args.min_confidence, "bin_s": BIN_S, "mean_smoothing_s": PEAK_SMOOTH_S,
                          "baseline": "mean over the fixation cross, per eye"},
                "gaze_min_confidence_drawn": gaze_conf,
                "gaze_min_confidence_for_offsets": 0.8,
                "screen_content_px": content_wh,
                "screen_content_how": f"automatic: last row brighter than {BLACK_LEVEL} in 5 sampled frames",
                "screen_gaze": str(screen_gaze_path) if screen_by_frame else None,
                "eye_labels": EYE_LABEL,
            },
            "sync_params": sync_params,
        },
    )  # fmt: skip
    print(f"wrote {record}")


if __name__ == "__main__":
    main()
