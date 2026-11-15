import cv2
import numpy as np
import pandas as pd

from pupil_analysis.surface import (
    CORNER_COLS,
    apply_h,
    display_quad,
    distort_px,
    fit_tag_model,
    frame_homographies,
    frame_poses,
    map_points,
    merge_tags,
    model_to_screen,
    undistort_px,
)

K = np.array([[794.0, 0, 633.0], [0, 793.5, 397.4], [0, 0, 1]])
D = np.array([-0.376, 0.164, 0.0001, 0.0001, 0.033, 0.082, -0.082, 0.145])
SCREEN = (2386.0, 1668.0)
# tag corners (screen px) just outside the display, as on the tablet bezel
MODEL = {
    t: np.array([[cx + 60, cy + 60], [cx + 60, cy - 60], [cx - 60, cy - 60], [cx - 60, cy + 60]], float)
    for t, (cx, cy) in {4: (2500, -120), 5: (2500, 1790), 6: (-110, -120), 7: (-110, 1790)}.items()
}


def project(pts_screen, rvec, tvec, dist=None):
    obj = np.c_[pts_screen, np.zeros(len(pts_screen))]
    return cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2)


def poses():
    rng = np.random.default_rng(0)
    for _ in range(5):
        rvec = np.array([np.pi + 0.3 * rng.standard_normal(), 0.2 * rng.standard_normal(), 0.1 * rng.standard_normal()])
        tvec = np.array([-1200 + 100 * rng.standard_normal(), -800 + 100 * rng.standard_normal(), 3000.0])
        yield rvec, tvec


def tag_table(frames):
    rows = []
    for f, (rvec, tvec) in enumerate(frames):
        for t, c in MODEL.items():
            rows.append([f, t, *project(c, rvec, tvec).ravel()])
    return pd.DataFrame(rows, columns=["world_index", "tag", *CORNER_COLS])


def test_undistort_roundtrip():
    pts = np.array([[100.0, 80.0], [640, 360], [1200, 650]])
    np.testing.assert_allclose(distort_px(undistort_px(pts, K, D), K, D), pts, atol=0.05)


def test_pose_maps_image_back_to_screen_even_with_two_tags():
    frames = list(poses())
    tags = tag_table(frames)  # already undistorted (projected without distortion)
    grid = np.array([[x, y] for x in (0, 1193, 2386) for y in (0, 834, 1668)], float)
    for keep in ([4, 5, 6, 7], [4, 5]):
        homs = frame_poses(tags[tags.tag.isin(keep)], MODEL, K).set_index("world_index")
        for f, (rvec, tvec) in enumerate(frames):
            np.testing.assert_allclose(apply_h(homs.at[f, "H"], project(grid, rvec, tvec)), grid, atol=0.5)


def test_tag_model_and_display_fix_gauge():
    frames = list(poses())
    model = fit_tag_model(tag_table(frames))  # some projective frame of the plane
    homs = frame_homographies(tag_table(frames), model).set_index("world_index")
    rvec, tvec = frames[0]
    disp_img = project(np.array([[0, 0], [SCREEN[0], 0], [SCREEN[0], SCREEN[1]], [0, SCREEN[1]]]), rvec, tvec)
    screen_model, _ = model_to_screen(model, apply_h(homs.at[0, "H"], disp_img), SCREEN)
    for t, corners in MODEL.items():
        np.testing.assert_allclose(screen_model[t], corners, atol=0.5)


def test_map_points_uses_distorted_norm_pos():
    rvec, tvec = next(poses())
    homs = frame_poses(tag_table([(rvec, tvec)]), MODEL, K)
    target = np.array([[700.0, 900.0]])
    px = project(target, rvec, tvec, D)  # what the (distorting) world camera sees
    df = pd.DataFrame({"x": px[:, 0] / 1280, "y": 1 - px[:, 1] / 720, "frame": [0]})
    out = map_points(df, homs, K, D)
    np.testing.assert_allclose(out[["sx", "sy"]].to_numpy(), target, atol=1.0)
    assert out.n_tags.iloc[0] == 4


def test_merge_prefers_player():
    p = pd.DataFrame([[1, 4, *range(8)]], columns=["world_index", "tag", *CORNER_COLS])
    o = pd.DataFrame([[1, 4, *range(8, 16)], [1, 5, *range(8)]], columns=["world_index", "tag", *CORNER_COLS])
    m = merge_tags(p, o)
    assert list(m.source) == ["player", "opencv"]
    assert m.corner_0_x.iloc[0] == 0


def test_display_quad_finds_rectangle_corners():
    img = np.zeros((720, 1280), np.uint8)
    corners = np.array([[300, 120], [980, 110], [990, 600], [290, 610]], np.int32)
    cv2.fillPoly(img, [corners], 255)
    q = display_quad(img, K, np.zeros(8))
    assert q is not None
    np.testing.assert_allclose(q, corners, atol=1.5)
