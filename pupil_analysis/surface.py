"""Map world-camera points (gaze) onto the tablet display, via the AprilTags on the tablet bezel.

Geometry:
- Each world frame: undistort the tag corners with the world camera's intrinsics, then fit a homography
  from that frame's (undistorted) image to a fixed model of the tag corners (`frame_homographies`).
- The tag model is expressed directly in screen-recording pixels: tag corners are fixed relative to the
  display, so one fixed transform (found from the display corners, `fit_display_corners`) carries the
  tag plane onto the screen recording.

Tag corners use Pupil Player's corner order (`marker_detections.csv`). Corners from OpenCV's detector
are reordered to match (`OPENCV_TO_PLAYER`, checked against Player on frames where both detect a tag).
"""

from pathlib import Path

import cv2
import msgpack
import numpy as np
import pandas as pd

from pupil_analysis.io import load_frame_lookup

CORNER_COLS = [f"corner_{k}_{a}" for k in range(4) for a in "xy"]
OPENCV_TO_PLAYER = [1, 0, 3, 2]  # player corner k = opencv corner OPENCV_TO_PLAYER[k]
WORLD_SIZE = (1280, 720)


def load_intrinsics(rec_dir: str | Path, size: tuple[int, int] = WORLD_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """World camera matrix and distortion coefficients from `world.intrinsics` (Pupil's radial model)."""
    d = msgpack.unpackb((Path(rec_dir) / "world.intrinsics").read_bytes(), raw=False)[str(size)]
    if d.get("cam_type") != "radial":
        raise ValueError(f"unsupported camera model {d.get('cam_type')!r}")
    return np.array(d["camera_matrix"], float), np.array(d["dist_coefs"], float).ravel()


def undistort_px(pts: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Distorted image px (N, 2) -> undistorted px (N, 2) in the same camera matrix."""
    pts = np.asarray(pts, float).reshape(-1, 1, 2)
    if len(pts) == 0:
        return pts.reshape(0, 2)
    # OpenCV's default 5 iterations do not converge near the image edges with this lens (up to 21 px off)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-9)
    return cv2.undistortPoints(pts, K, D, None, K, criteria=crit).reshape(-1, 2)


def distort_px(pts: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Inverse of `undistort_px`."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    norm = np.c_[(pts - K[:2, 2]) / [K[0, 0], K[1, 1]], np.ones(len(pts))]
    out, _ = cv2.projectPoints(norm.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    return out.reshape(-1, 2)


def detect_tags_opencv(rec_dir: str | Path, tag_ids=(4, 5, 6, 7)) -> pd.DataFrame:
    """Detect AprilTags (tag36h11) in every world frame with OpenCV. Columns: `world_index` (Player frame),
    `tag`, `CORNER_COLS` (Player corner order, distorted image px). Reads the video sequentially."""
    rec_dir = Path(rec_dir)
    video_frame = load_frame_lookup(rec_dir).video_frame.to_numpy()
    player_of = {int(v): i for i, v in enumerate(video_frame) if v >= 0}
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)
    rows = []
    cap = cv2.VideoCapture(str(rec_dir / "world.mp4"))
    try:
        v = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if ids is not None and v in player_of:
                for c, i in zip(corners, ids.ravel()):
                    if int(i) in tag_ids:
                        rows.append([player_of[v], int(i), *c[0][OPENCV_TO_PLAYER].ravel()])
            v += 1
    finally:
        cap.release()
    return pd.DataFrame(rows, columns=["world_index", "tag", *CORNER_COLS])


def load_player_tags(rec_dir: str | Path, tag_ids=(4, 5, 6, 7)) -> pd.DataFrame:
    """Tags from the newest Pupil Player export (`surfaces/marker_detections.csv`), same columns."""
    path = max(Path(rec_dir).glob("exports/*/surfaces/marker_detections.csv"))
    m = pd.read_csv(path)
    m["tag"] = m.marker_uid.str.rsplit(":", n=1).str[-1].astype(int)
    return m[m.tag.isin(tag_ids)][["world_index", "tag", *CORNER_COLS]].reset_index(drop=True)


def merge_tags(player: pd.DataFrame, opencv: pd.DataFrame) -> pd.DataFrame:
    """Player's detection where it has one, OpenCV's for the (frame, tag) pairs Player missed. Adds `source`."""
    p = player.assign(source="player")
    have = pd.MultiIndex.from_frame(p[["world_index", "tag"]])
    extra = opencv[~pd.MultiIndex.from_frame(opencv[["world_index", "tag"]]).isin(have)].assign(source="opencv")
    return pd.concat([p, extra]).sort_values(["world_index", "tag"]).reset_index(drop=True)


def undistorted_corners(tags: pd.DataFrame, K: np.ndarray, D: np.ndarray) -> pd.DataFrame:
    """Same table with corner columns undistorted."""
    out = tags.copy()
    out[CORNER_COLS] = undistort_px(tags[CORNER_COLS].to_numpy().reshape(-1, 2), K, D).reshape(-1, 8)
    return out


def _frame_points(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(tag ids repeated per corner, corner points) for one frame."""
    return np.repeat(group.tag.to_numpy(), 4), group[CORNER_COLS].to_numpy().reshape(-1, 2)


def fit_tag_model(tags_ud: pd.DataFrame, n_iter: int = 4) -> dict[int, np.ndarray]:
    """Fixed tag-corner coordinates in a common plane frame, from frames that see all tags.

    Starts from one frame's undistorted corners and repeatedly maps every all-tag frame onto the current
    model, taking the per-corner median. The result is defined up to a homography; `fit_display_corners`
    fixes it by mapping the display onto screen-recording pixels.
    """
    tag_ids = sorted(tags_ud.tag.unique())
    full = [g for _, g in tags_ud.groupby("world_index") if sorted(g.tag) == tag_ids]
    if not full:
        raise ValueError("no frame shows every tag")
    order = np.argsort([g.tag.to_numpy() for g in full], axis=1)
    frames = [g.iloc[o][CORNER_COLS].to_numpy().reshape(-1, 2) for g, o in zip(full, order)]
    model = frames[len(frames) // 2]
    for _ in range(n_iter):
        mapped = []
        for pts in frames:
            H, _ = cv2.findHomography(pts, model, 0)
            if H is not None:
                mapped.append(cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2))
        model = np.median(mapped, axis=0)
    return {t: model[4 * k : 4 * k + 4] for k, t in enumerate(tag_ids)}


def frame_homographies(tags_ud: pd.DataFrame, model: dict[int, np.ndarray], min_tags: int = 2) -> pd.DataFrame:
    """Per world frame: homography from undistorted image px to model coordinates.

    Columns: `world_index`, `n_tags`, `tags` (e.g. "457"), `rms` (model-unit reprojection error of the
    tag corners), `H` (3x3 array). Frames with fewer than `min_tags` tags are left out.
    """
    rows = []
    for _, g in tags_ud.groupby("world_index"):
        if len(g) < min_tags:
            continue
        _, src = _frame_points(g)
        dst = np.concatenate([model[t] for t in g.tag])
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            continue
        err = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2) - dst
        rows.append({"world_index": int(g.world_index.iloc[0]), "n_tags": len(g), "tags": "".join(map(str, sorted(g.tag))),
                     "rms": float(np.sqrt((err**2).sum(1).mean())), "H": H})  # fmt: skip
    return pd.DataFrame(rows)


def frame_poses(tags_ud: pd.DataFrame, model: dict[int, np.ndarray], K: np.ndarray, min_tags: int = 2) -> pd.DataFrame:
    """Like `frame_homographies`, but via the camera pose (PnP on the calibrated camera).

    `model` must be a metric frame of the plane (screen-recording px are: the display's pixels are square).
    A pose has 6 degrees of freedom vs a homography's 8, so it stays well constrained when only two
    neighbouring tags are visible, where a free homography extrapolates badly across the display.
    `H` maps undistorted image px to model coordinates; `rms` is the tag-corner reprojection error in
    world-image px.
    """
    rows = []
    for _, g in tags_ud.groupby("world_index"):
        if len(g) < min_tags:
            continue
        _, img = _frame_points(g)
        obj = np.c_[np.concatenate([model[t] for t in g.tag]), np.zeros(4 * len(g))]
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        R = cv2.Rodrigues(rvec)[0]
        H_plane_to_img = K @ np.c_[R[:, :2], tvec.ravel()]
        rows.append({"world_index": int(g.world_index.iloc[0]), "n_tags": len(g), "tags": "".join(map(str, sorted(g.tag))),
                     "rms": float(np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(1).mean())),
                     "H": np.linalg.inv(H_plane_to_img)})  # fmt: skip
    return pd.DataFrame(rows)


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def display_quad(gray: np.ndarray, K: np.ndarray, D: np.ndarray, trim: float = 0.15) -> np.ndarray | None:
    """Corners (TL, TR, BR, BL; undistorted px) of the bright display in a world frame showing a white screen.

    Otsu-threshold, take the largest bright contour, undistort it, split it into 4 sides at the polygon's
    corners, fit a line to the middle (1 - 2*trim) of each side (skipping the rounded corners), and
    intersect neighbouring lines. Returns None if the contour is not a quadrilateral.
    """
    _, bw = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * gray.size:
        return None
    approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(approx) != 4:
        return None
    pts = c.reshape(-1, 2)
    idx = sorted(int(np.argmin(((pts - a) ** 2).sum(1))) for a in approx)
    lines = []
    for k in range(4):
        a, b = idx[k], idx[(k + 1) % 4]
        side = pts[a:b] if b > a else np.r_[pts[a:], pts[:b]]
        n = len(side)
        side = side[int(trim * n) : int((1 - trim) * n)]
        if len(side) < 10:
            return None
        lines.append(cv2.fitLine(undistort_px(side, K, D).astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
    corners = []
    for k in range(4):
        (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = lines[k - 1], lines[k]
        A = np.array([[vx1, -vx2], [vy1, -vy2]])
        s = np.linalg.solve(A, [x2 - x1, y2 - y1])[0]
        corners.append([x1 + s * vx1, y1 + s * vy1])
    corners = np.array(corners)
    # order TL, TR, BR, BL (image y down)
    sm, df = corners.sum(1), corners[:, 0] - corners[:, 1]
    return corners[[np.argmin(sm), np.argmax(df), np.argmax(sm), np.argmin(df)]]


def model_to_screen(model: dict[int, np.ndarray], display_model: np.ndarray, screen_wh: tuple[float, float]):
    """Re-express the tag model in screen-recording px, given the display corners (TL, TR, BR, BL) in model
    coordinates and the recording's content size. Returns (tag model in screen px, the plane->screen H)."""
    w, h = screen_wh
    G, _ = cv2.findHomography(display_model, np.array([[0, 0], [w, 0], [w, h], [0, h]], float), 0)
    return {t: apply_h(G, c) for t, c in model.items()}, G


def map_points(
    df: pd.DataFrame, homs: pd.DataFrame, K: np.ndarray, D: np.ndarray, size: tuple[int, int] = WORLD_SIZE
) -> pd.DataFrame:
    """Map normalized world points (`x`, `y`, bottom-left origin) in frame `frame` to screen px.

    Adds `sx`, `sy` (screen-recording px, top-left origin), `n_tags`, `tags`. Points in frames without a
    homography get NaN.
    """
    w, h = size
    out = df.copy()
    px = np.c_[df.x.to_numpy() * w, (1 - df.y.to_numpy()) * h]
    ud = undistort_px(px, K, D)
    by = homs.set_index("world_index")
    sx = np.full(len(df), np.nan)
    sy = np.full(len(df), np.nan)
    frames = df.frame.to_numpy()
    for f in np.intersect1d(np.unique(frames), by.index):
        sel = frames == f
        sx[sel], sy[sel] = apply_h(by.at[f, "H"], ud[sel]).T
    out["sx"], out["sy"] = sx, sy
    out["n_tags"] = by.n_tags.reindex(frames).to_numpy()
    out["tags"] = by.tags.reindex(frames).to_numpy()
    return out
