"""Telestration on endoscopy video (Big Data assignment 2-4).

The surgeon pauses on a frame, clicks 2 points (line) or 3 points (polyline A->B->C),
and the drawing has to stay on the same tissue while the endoscope moves.

Three trackers solve the "camera moving problem" and are compared on the same video:
  A) Pyramidal Lucas-Kanade on the clicked points       (Lecture 4, slides 80-88)
  B) ORB + ratio test + RANSAC homography (global motion) (Lecture 4, slides 51, 61-69, 109, 115)
  C) Phase correlation, translation only                (Lecture 4, slides 171-175)

Preprocessing (CLAHE, specular-highlight mask built with morphology) is applied only to the
grayscale copy that the trackers see; the video is always shown in its original colours.

Run:  streamlit run telestration_app.py
"""

import os
import tempfile
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------

# Everything is processed at 640 px width: endoscopy video is often HD, and tracking a few points
# does not need that resolution. It keeps a 5 s clip at a few seconds of processing time.
PROC_WIDTH = 640
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v")
DEFAULT_VIDEO_DIR = os.environ.get(
    "TELESTRATION_VIDEO_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
)

TRACKERS = ["LK", "ORB", "PC"]
TRACKER_LABELS = {
    "LK": "A) Lucas-Kanade (clicked points)",
    "ORB": "B) ORB + RANSAC homography",
    "PC": "C) Phase correlation (translation)",
}
# One colour per tracker, identical in the video (BGR) and in the plots (hex).
TRACKER_HEX = {"LK": "#00B8D4", "ORB": "#2EBD4E", "PC": "#F5A300"}
TRACKER_BGR = {"LK": (212, 184, 0), "ORB": (78, 189, 46), "PC": (0, 163, 245)}

# Lucas-Kanade parameters. winSize 31: endoscopic tissue has little texture, so a larger window
# collects enough gradient information (lambda1, lambda2 both large, slide 86). maxLevel 3 builds a
# 4-level Gaussian pyramid so motions far larger than the window are found coarse-to-fine (slide 88).
# The criteria is the iterative refinement stop rule (slide 87).
LK_PARAMS = dict(
    winSize=(31, 31),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
LOWE_RATIO = 0.75  # ratio distance threshold (slide 63, Lowe 2004 slide 69)
RANSAC_REPROJ_PX = 3.0  # a match is an inlier if the homography maps it within 3 px
FAST_THRESHOLD = 20  # used for the "number of good features" statistic (slide 115)


@dataclass(frozen=True)
class Prep:
    """Preprocessing settings. Frozen so it can be compared and used as a cache key."""

    clahe: bool = False
    clip_limit: float = 2.0
    tile: int = 8
    glare: bool = False
    glare_thresh: int = 220
    glare_dilate: int = 5
    border: bool = False


NO_PREP = Prep()


# ----------------------------------------------------------------------------------------------
# Video input / output
# ----------------------------------------------------------------------------------------------


def resize_to_width(frame, width=PROC_WIDTH):
    """Downscale to the processing width; height is kept even because H.264 (yuv420p) needs it."""
    h, w = frame.shape[:2]
    new_h = int(round(h * width / w / 2.0) * 2)
    if (w, h) == (width, new_h):
        return frame
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # Some containers report 0 or nonsense fps; 25 is the usual endoscopy tower frame rate.
    if not fps or fps != fps or fps > 240:
        fps = 25.0
    return float(fps), n, w, h


def read_clip(path, start, n_frames):
    """Read n_frames consecutive frames from `start`, resized to PROC_WIDTH (BGR, uint8)."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(resize_to_width(frame))
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 2, 2, 3), np.uint8)


def encode_h264(frames_rgb, fps):
    """Encode RGB frames to an H.264 mp4 that browsers can play.

    OpenCV's VideoWriter usually only has the 'mp4v' (MPEG-4 part 2) codec, which Chrome/Safari
    refuse to play. imageio-ffmpeg ships its own ffmpeg binary with libx264.
    Returns the mp4 bytes, or None if encoding failed (the app then falls back to the frame slider).
    """
    try:
        import imageio_ffmpeg

        h, w = frames_rgb[0].shape[:2]
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        writer = imageio_ffmpeg.write_frames(
            path,
            (w, h),
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            quality=8,
            output_params=["-movflags", "+faststart"],
        )
        writer.send(None)  # start the generator
        for f in frames_rgb:
            writer.send(np.ascontiguousarray(f))
        writer.close()
        with open(path, "rb") as fh:
            data = fh.read()
        os.remove(path)
        return data
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------------------------


def specular_mask(gray, thresh, dilate_px):
    """255 where the image shows a specular highlight (glare) plus a safety margin.

    Glare is a mirror reflection of the endoscope's light source. It moves with the camera/light,
    not with the tissue, so every feature found on it is a false track.
      1. threshold: glare is (almost) saturated white
      2. dilation (Lecture 2 slide 83): grows each blob so its bright rim/halo, which has strong
         gradients that corner detectors love, is excluded as well, and merges nearby speckles.
    We deliberately do NOT use an opening (slides 86-87) here: on wet fat the glare consists of
    1-3 px speckles, and an opening would delete exactly those as "noise".
    """
    _, bright = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        bright = cv2.dilate(bright, k)
    return bright


def border_mask(gray):
    """255 on the black frame around the endoscope image (and on text overlays sitting on it).

    That frame is fixed to the camera. Features on its edge never move, so they pull the
    homography and the phase correlation towards "no motion". Dilated so the edge itself is
    excluded too.
    """
    _, dark = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))


def preprocess(frame_bgr, prep):
    """Return (gray for tracking, valid mask 255 = usable, excluded mask 255 = glare/border)."""
    raw = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = raw
    if prep.clahe:
        # CLAHE = Contrast Limited Adaptive Histogram Equalization (not in the slides; general
        # theory). Equalizes the histogram per tile, so faint tissue texture in dark and bright
        # parts of the image both get stretched -> more corners with large lambda1/lambda2.
        # The clip limit caps the stretch so noise in flat areas is not blown up.
        clahe = cv2.createCLAHE(clipLimit=prep.clip_limit, tileGridSize=(prep.tile, prep.tile))
        gray = clahe.apply(raw)
    excluded = np.zeros_like(raw)
    # Masks are computed on the raw image: CLAHE would change what counts as "saturated".
    if prep.glare:
        glare = specular_mask(raw, prep.glare_thresh, prep.glare_dilate)
        excluded |= glare
        # The mask stops ORB from detecting features on glare, but LK and phase correlation work
        # on raw pixels in a window and cannot take a mask. A static glare blob inside the LK
        # window pulls the tracked point towards it. Inpainting fills the glare from the
        # surrounding tissue, so no tracker "sees" the reflection at all.
        gray = cv2.inpaint(gray, glare, 3, cv2.INPAINT_TELEA)
    if prep.border:
        excluded |= border_mask(raw)
    return gray, cv2.bitwise_not(excluded), excluded


_FAST = cv2.FastFeatureDetector_create(threshold=FAST_THRESHOLD, nonmaxSuppression=True)


def count_fast(gray, mask=None):
    return len(_FAST.detect(gray, mask))


# ----------------------------------------------------------------------------------------------
# Trackers
# ----------------------------------------------------------------------------------------------


def plausible_homography(H):
    """Reject frame-to-frame homographies that cannot be real camera motion at 25 fps.

    Between two consecutive frames the scale changes only a little and the perspective terms are
    tiny. A wild H means RANSAC found a consensus on a wrong set (e.g. glare, smoke).
    """
    if H is None or not np.all(np.isfinite(H)):
        return False
    det = np.linalg.det(H[:2, :2])
    return 0.7 < det < 1.4 and abs(H[2, 0]) < 0.002 and abs(H[2, 1]) < 0.002


def run_trackers(frames, points, prep, fb_thresh=4.0):
    """Track the drawing through the clip with all three trackers.

    frames: (T, H, W, 3) BGR uint8, points: list of (x, y) on frame 0.
    Returns tracks {name: (T, P, 2)}, lk_lost (T, P) bool, per-frame stats DataFrame.
    """
    T = len(frames)
    pts0 = np.array(points, np.float32).reshape(-1, 1, 2)
    P = len(points)

    orb = cv2.ORB_create(nfeatures=1000, fastThreshold=12)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)  # Hamming distance: ORB descriptors are binary (BRIEF)
    h, w = frames.shape[1:3]
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)  # avoids edge effects in the FFT

    tracks = {k: np.zeros((T, P, 2), np.float32) for k in TRACKERS}
    for k in TRACKERS:
        tracks[k][0] = pts0.reshape(-1, 2)
    lk_lost = np.zeros((T, P), bool)
    rows = []

    t0 = time.perf_counter()
    prev_gray, prev_mask, prev_excl = preprocess(frames[0], prep)
    prep_ms0 = (time.perf_counter() - t0) * 1000

    def pc_input(gray, excl):
        # Phase correlation cannot take a mask, so excluded pixels are set to the mean grey
        # value: then they carry no structure that could produce a (static) correlation peak.
        f = gray.astype(np.float32)
        if excl.any():
            f[excl > 0] = f[excl == 0].mean() if (excl == 0).any() else 0
        return f

    lk_pts = pts0.copy()
    lk_alive = np.ones(P, bool)
    H_cum = np.eye(3)
    pc_offset = np.zeros(2, np.float32)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, prev_mask)
    prev_pc = pc_input(prev_gray, prev_excl)

    def feature_counts(frame_bgr, gray, mask):
        # Corners on the raw image, corners the trackers actually get, and raw corners that sit
        # on glare/border (counted on the raw image, because inpainting already removed them).
        raw = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        n_raw = count_fast(raw)
        return dict(feat_raw=n_raw, feat_prep=count_fast(gray, mask),
                    feat_excluded=n_raw - count_fast(raw, mask))

    rows.append(
        dict(
            frame=0,
            **feature_counts(frames[0], prev_gray, prev_mask),
            lk_fb_err=np.nan,
            lk_lost=0,
            orb_matches=np.nan,
            orb_inlier_ratio=np.nan,
            orb_ok=True,
            pc_response=np.nan,
            ms_prep=prep_ms0,
            ms_LK=np.nan,
            ms_ORB=np.nan,
            ms_PC=np.nan,
        )
    )

    for t in range(1, T):
        t_start = time.perf_counter()
        gray, mask, excl = preprocess(frames[t], prep)
        ms_prep = (time.perf_counter() - t_start) * 1000

        # --- A) Pyramidal Lucas-Kanade directly on the clicked points -------------------------
        t_start = time.perf_counter()
        fb_mean = np.nan
        if lk_alive.any():
            p0 = lk_pts[lk_alive]
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
            # Forward-backward check: track the result back to the previous frame. If brightness
            # constancy / small motion held, we land where we started. A large round-trip error
            # means the LK assumptions were broken (fast motion, smoke, blood, glare).
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **LK_PARAMS)
            fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
            ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < fb_thresh)
            fb_mean = float(fb.mean())
            idx = np.flatnonzero(lk_alive)
            lk_pts[idx[ok]] = p1[ok]
            # A lost point is frozen at its last good position: LK has no way to recover it.
            lk_alive[idx[~ok]] = False
        tracks["LK"][t] = lk_pts.reshape(-1, 2)
        lk_lost[t] = ~lk_alive
        ms_lk = (time.perf_counter() - t_start) * 1000

        # --- B) ORB + ratio test + RANSAC homography, accumulated frame to frame ---------------
        t_start = time.perf_counter()
        kp, des = orb.detectAndCompute(gray, mask)
        n_good, inlier_ratio, orb_ok = 0, np.nan, False
        if prev_des is not None and des is not None and len(prev_kp) >= 8 and len(kp) >= 8:
            knn = bf.knnMatch(prev_des, des, k=2)
            # Lowe ratio test: keep a match only if it is clearly better than the 2nd best.
            # Ambiguous matches (repetitive mucosa) get ratio close to 1 and are dropped.
            good = [p[0] for p in knn if len(p) == 2 and p[0].distance < LOWE_RATIO * p[1].distance]
            n_good = len(good)
            if n_good >= 8:
                src = np.float32([prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                # RANSAC: fit H to random 4-match samples, keep the one most matches agree with.
                # Wrong matches and independently moving things (instruments) become outliers.
                H, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_PX)
                if plausible_homography(H):
                    H_cum = H @ H_cum  # chain: frame0 -> t-1 -> t
                    inlier_ratio = float(inl.sum()) / n_good
                    orb_ok = True
        # If no good H was found the drawing simply stays where it was (assume no motion).
        tracks["ORB"][t] = cv2.perspectiveTransform(pts0, H_cum).reshape(-1, 2)
        prev_kp, prev_des = kp, des
        ms_orb = (time.perf_counter() - t_start) * 1000

        # --- C) Phase correlation: one global translation per frame ---------------------------
        t_start = time.perf_counter()
        cur_pc = pc_input(gray, excl)
        (dx, dy), response = cv2.phaseCorrelate(prev_pc, cur_pc, hann)
        pc_offset += np.float32([dx, dy])
        tracks["PC"][t] = (pts0.reshape(-1, 2) + pc_offset)
        prev_pc = cur_pc
        ms_pc = (time.perf_counter() - t_start) * 1000

        rows.append(
            dict(
                frame=t,
                **feature_counts(frames[t], gray, mask),
                lk_fb_err=fb_mean,
                lk_lost=int((~lk_alive).sum()),
                orb_matches=n_good,
                orb_inlier_ratio=inlier_ratio,
                orb_ok=orb_ok,
                pc_response=float(response),
                ms_prep=ms_prep,
                ms_LK=ms_lk,
                ms_ORB=ms_orb,
                ms_PC=ms_pc,
            )
        )
        prev_gray, prev_mask, prev_excl = gray, mask, excl

    return tracks, lk_lost, pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------------------


def draw_points(img, pts, color, closed=False, lost=None, labels=True):
    """Draw the telestration line A-B(-C) with round end points; lost points get an X."""
    p = np.round(pts).astype(np.int32)
    cv2.polylines(img, [p.reshape(-1, 1, 2)], closed, color, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(p):
        if lost is not None and lost[i]:
            cv2.drawMarker(img, (int(x), int(y)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        else:
            cv2.circle(img, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 1, cv2.LINE_AA)
        if labels:
            cv2.putText(img, "ABC"[i], (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)


def render_frame(frame_bgr, t, fps, tracks, lk_lost, show, show_static=True):
    """Overlay all selected trackers on one frame and return it as RGB."""
    img = frame_bgr.copy()
    if show_static:
        # Where the drawing would stay without any tracking: makes the camera motion visible.
        static = tracks["LK"][0]
        cv2.polylines(img, [np.round(static).astype(np.int32).reshape(-1, 1, 2)], False,
                      (255, 255, 255), 1, cv2.LINE_AA)
    for name in show:
        draw_points(img, tracks[name][t], TRACKER_BGR[name],
                    lost=lk_lost[t] if name == "LK" else None, labels=(name == show[0]))

    # Legend box, semi-transparent so the tissue stays visible.
    entries = [(TRACKER_LABELS[n], TRACKER_BGR[n]) for n in show]
    if show_static:
        entries.append(("no tracking (start position)", (255, 255, 255)))
    box_h = 8 + 18 * len(entries)
    overlay = img.copy()
    cv2.rectangle(overlay, (6, 6), (300, 6 + box_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    for i, (label, color) in enumerate(entries):
        y = 22 + 18 * i
        cv2.line(img, (14, y - 4), (34, y - 4), color, 3, cv2.LINE_AA)
        cv2.putText(img, label, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    h = img.shape[0]
    cv2.putText(img, f"frame {t}  t = {t / fps:4.2f} s", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preview_panel(frame_bgr, prep):
    """Before/after images: raw grayscale with FAST corners vs preprocessed + mask + corners."""
    raw = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray, mask, excl = preprocess(frame_bgr, prep)
    before = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    for kp in _FAST.detect(raw, None):
        cv2.circle(before, tuple(int(v) for v in kp.pt), 2, (0, 255, 0), -1)
    after = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    after[excl > 0] = (0.5 * after[excl > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    for kp in _FAST.detect(gray, mask):
        cv2.circle(after, tuple(int(v) for v in kp.pt), 2, (0, 255, 0), -1)
    return (cv2.cvtColor(before, cv2.COLOR_BGR2RGB), cv2.cvtColor(after, cv2.COLOR_BGR2RGB),
            count_fast(raw), count_fast(gray, mask), float((excl > 0).mean() * 100))


# ----------------------------------------------------------------------------------------------
# Statistics plots
# ----------------------------------------------------------------------------------------------


def _layout(fig, title, ytitle, xtitle="time (s)"):
    fig.update_layout(title=title, height=330, margin=dict(l=10, r=10, t=45, b=10),
                      legend=dict(orientation="h", y=-0.25), xaxis_title=xtitle,
                      yaxis_title=ytitle)
    return fig


def displacement(tracks, name):
    """Mean distance (px) of the drawing's points from where they were clicked."""
    return np.linalg.norm(tracks[name] - tracks[name][0], axis=2).mean(axis=1)


def plot_features(df, fps):
    t = df.frame / fps
    fig = go.Figure()
    fig.add_scatter(x=t, y=df.feat_raw, name="raw grayscale", line=dict(color="#8C8C8C"))
    fig.add_scatter(x=t, y=df.feat_prep, name="preprocessed (CLAHE + mask)",
                    line=dict(color="#2EBD4E"))
    fig.add_scatter(x=t, y=df.feat_excluded, name="rejected on glare / border",
                    line=dict(color="#E5484D", dash="dot"))
    return _layout(fig, "Good features (FAST corners) per frame", "corners")


def plot_lk(df, df_base, fps):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.08)
    t = df.frame / fps
    fig.add_scatter(x=t, y=df.lk_fb_err, name="FB error, preprocessed",
                    line=dict(color=TRACKER_HEX["LK"]), row=1, col=1)
    fig.add_scatter(x=t, y=df_base.lk_fb_err, name="FB error, no preprocessing",
                    line=dict(color="#8C8C8C", dash="dot"), row=1, col=1)
    fig.add_scatter(x=t, y=df.lk_lost, name="points lost, preprocessed",
                    line=dict(color=TRACKER_HEX["LK"], shape="hv"), row=2, col=1)
    fig.add_scatter(x=t, y=df_base.lk_lost, name="points lost, no preprocessing",
                    line=dict(color="#8C8C8C", dash="dot", shape="hv"), row=2, col=1)
    fig.update_yaxes(title_text="FB error (px)", row=1, col=1)
    fig.update_yaxes(title_text="lost", row=2, col=1)
    fig.update_layout(title="A) LK: forward-backward error, lost points",
                      height=380, margin=dict(l=10, r=10, t=45, b=10),
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    return fig


def plot_orb(df, df_base, fps):
    fig = go.Figure()
    fig.add_scatter(x=df.frame / fps, y=df.orb_inlier_ratio, name="preprocessed",
                    line=dict(color=TRACKER_HEX["ORB"]))
    fig.add_scatter(x=df_base.frame / fps, y=df_base.orb_inlier_ratio, name="no preprocessing",
                    line=dict(color="#8C8C8C", dash="dot"))
    fig.update_yaxes(range=[0, 1.02])
    return _layout(fig, "B) ORB: RANSAC inlier ratio per frame (gaps = no valid homography)",
                   "inliers / matches")


def plot_displacement(tracks, fps):
    fig = go.Figure()
    t = np.arange(len(tracks["LK"])) / fps
    for name in TRACKERS:
        fig.add_scatter(x=t, y=displacement(tracks, name), name=TRACKER_LABELS[name],
                        line=dict(color=TRACKER_HEX[name]))
    lk_vs_orb = np.linalg.norm(tracks["LK"] - tracks["ORB"], axis=2).mean(axis=1)
    fig.add_scatter(x=t, y=lk_vs_orb, name="disagreement LK vs ORB",
                    line=dict(color="#E5484D", dash="dot"))
    return _layout(fig, "Displacement of the drawing from its start position", "pixels")


def plot_timing(df, fps):
    cols = ["ms_prep", "ms_LK", "ms_ORB", "ms_PC"]
    names = ["Preprocessing", TRACKER_LABELS["LK"], TRACKER_LABELS["ORB"], TRACKER_LABELS["PC"]]
    colors = ["#8C8C8C", TRACKER_HEX["LK"], TRACKER_HEX["ORB"], TRACKER_HEX["PC"]]
    means = [df[c].mean() for c in cols]
    fig = go.Figure(go.Bar(x=names, y=means, marker_color=colors,
                           text=[f"{m:.1f} ms" for m in means], textposition="outside"))
    fig.add_hline(y=1000 / fps, line_dash="dot", line_color="#E5484D",
                  annotation_text=f"real-time budget at {fps:.0f} fps")
    fig.update_yaxes(range=[0, max(max(means), 1000 / fps) * 1.25])
    return _layout(fig, "Processing time per frame (mean)", "ms", xtitle="")


# ----------------------------------------------------------------------------------------------
# Discussion
# ----------------------------------------------------------------------------------------------


def summary_numbers(tracks, lk_lost, df, df_base):
    T = len(df)
    lk_vs_orb = np.linalg.norm(tracks["LK"] - tracks["ORB"], axis=2).mean(axis=1)
    pc_vs_orb = np.linalg.norm(tracks["PC"] - tracks["ORB"], axis=2).mean(axis=1)
    return dict(
        lk_kept=int((~lk_lost[-1]).sum()),
        n_pts=lk_lost.shape[1],
        lk_kept_base=int(len(lk_lost[-1]) - df_base.lk_lost.iloc[-1]),
        lk_fb=float(np.nanmean(df.lk_fb_err)),
        orb_ok=float(df.orb_ok.iloc[1:].mean()) if T > 1 else 1.0,
        orb_inl=float(np.nanmean(df.orb_inlier_ratio)),
        orb_inl_base=float(np.nanmean(df_base.orb_inlier_ratio)),
        feat_raw=float(df.feat_raw.mean()),
        feat_prep=float(df.feat_prep.mean()),
        lk_vs_orb_end=float(lk_vs_orb[-1]),
        pc_vs_orb_end=float(pc_vs_orb[-1]),
        motion=float(displacement(tracks, "ORB")[-1]),
        ms={k: float(df[f"ms_{k}"].mean()) for k in TRACKERS},
    )


def discussion_markdown(s):
    lk_status = (
        f"kept **{s['lk_kept']}/{s['n_pts']}** points through the clip"
        if s["lk_kept"] == s["n_pts"]
        else f"lost **{s['n_pts'] - s['lk_kept']}/{s['n_pts']}** points (frozen, red X in the video)"
    )
    return f"""
**What happened on this clip** (preprocessed run; the drawing moved about **{s['motion']:.0f} px**
according to ORB):

- **A) Lucas-Kanade** {lk_status}; mean forward-backward error {s['lk_fb']:.2f} px
  (without preprocessing it kept {s['lk_kept_base']}/{s['n_pts']}).
- **B) ORB + RANSAC** found a valid homography in **{s['orb_ok'] * 100:.0f} %** of the frames with a
  mean inlier ratio of **{s['orb_inl']:.2f}** (no preprocessing: {s['orb_inl_base']:.2f}).
- **C) Phase correlation** ends **{s['pc_vs_orb_end']:.0f} px** from ORB, LK ends
  **{s['lk_vs_orb_end']:.0f} px** from ORB.
- Preprocessing changed the number of usable corners from {s['feat_raw']:.0f} to
  {s['feat_prep']:.0f} per frame, after removing corners on glare and border.
- Time per frame: LK {s['ms']['LK']:.1f} ms, ORB {s['ms']['ORB']:.1f} ms,
  phase correlation {s['ms']['PC']:.1f} ms.

**Why, in terms of the lecture**

- **Lucas-Kanade** (Lecture 4, slides 80-88) assumes *brightness constancy*, *small motion* and
  *spatial coherence*. It tracks only the 2-3 clicked pixels, so it depends completely on those
  pixels having texture (both eigenvalues λ1, λ2 large, slide 86). Smooth mucosa, fast camera moves,
  smoke, blood or a passing instrument break the assumptions and the point is lost or slides along an
  edge (λ1 large, λ2 small). The Gaussian pyramid (slide 88) helps with larger motion, the
  forward-backward check tells us *when* it failed. It is the cheapest tracker.
- **ORB + homography** (slides 61-69, 109, 115) uses hundreds of features from the whole image, the
  Lowe ratio test removes ambiguous matches, and RANSAC (slide 51) rejects wrong matches and things
  that move on their own (instruments). It is the most robust against losing the drawing because it
  does not depend on the clicked pixels. But a homography assumes a *planar* scene (or pure camera
  rotation): close to deep, curved anatomy the drawing is placed correctly only near the dominant
  plane. Chaining frame-to-frame homographies also accumulates **drift** (slide 82).
- **Phase correlation** (slides 171-175) is fast and robust to noise and illumination, but it only
  estimates one global **translation**: zoom (moving the scope in/out) and rotation are not modelled,
  so the drawing shifts correctly but does not scale or rotate with the tissue.
- **Preprocessing**: CLAHE is not in the slides; it is standard contrast-limited histogram
  equalization that lifts faint texture so detectors find more corners. The specular-highlight mask
  (threshold → dilation, morphology from Lecture 2, then inpainting) is medically motivated: glare
  moves with the light source on the scope tip, not with the tissue, so features on it are false
  tracks, and a glare blob inside the LK window drags the tracked point along with the camera.
  No opening is used: on wet fat the glare is 1-3 px speckles that an opening would delete.

**Conclusion:** for telestration where the drawing must survive a few seconds of camera movement, the
global ORB + RANSAC homography is the most reliable choice; LK on the clicked points is the cheapest
and most accurate *while* the points have texture, and phase correlation is a fast fallback for pure
panning.

**Future work, YOLO** (slides 186-190): YOLOv11 detects *objects* such as instruments, not tissue, so
it does not solve the camera-motion problem directly. It could be used to (1) mask instruments out
of the ORB features, and (2) anchor a drawing to the instrument tip instead of to the tissue.
"""


# ----------------------------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------------------------


def pick_video():
    """Sidebar: choose a video from a folder or upload one. Returns a file path or None."""
    st.sidebar.header("1 · Video")
    folder = st.sidebar.text_input("Video folder", DEFAULT_VIDEO_DIR)
    folder = os.path.expanduser(folder)
    files = []
    if os.path.isdir(folder):
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTS))
    upload = st.sidebar.file_uploader("…or upload a video", type=[e[1:] for e in VIDEO_EXTS])
    if upload is not None:
        # Uploaded files only live in memory; OpenCV needs a path, so write a temp copy once.
        key = f"upload_{upload.name}_{upload.size}"
        if key not in st.session_state:
            suffix = os.path.splitext(upload.name)[1]
            fd, path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as fh:
                fh.write(upload.getbuffer())
            st.session_state[key] = path
        return st.session_state[key], upload.name
    if not files:
        st.sidebar.warning(f"No videos found in `{folder}`.")
        return None, None
    name = st.sidebar.selectbox("Video", files)
    return os.path.join(folder, name), name


@st.cache_resource(show_spinner="Reading clip…", max_entries=3)
def cached_clip(path, start, n_frames):
    # cache_resource returns the same array object (no copy), which is fine: it is never modified.
    return read_clip(path, start, n_frames)


@st.cache_data(show_spinner=False)
def cached_info(path):
    return video_info(path)


def click_points(frame_bgr, n_needed, state_key):
    """Show the start frame; each click adds a point. Returns the list of points."""
    from streamlit_image_coordinates import streamlit_image_coordinates

    if st.session_state.get("points_key") != state_key:
        st.session_state.points_key = state_key
        st.session_state.points = []
    pts = st.session_state.points[:n_needed]

    img = frame_bgr.copy()
    if pts:
        draw_points(img, np.array(pts, np.float32), (0, 255, 255))
    # The widget key includes the number of points: after each click a fresh widget is created,
    # so the same click is never counted twice when Streamlit reruns the script.
    click = streamlit_image_coordinates(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=img.shape[1], key=f"click_{state_key}_{len(pts)}"
    )
    if click is not None and len(pts) < n_needed:
        # Scale in case the browser displayed the image at a different size.
        sx = img.shape[1] / click.get("width", img.shape[1])
        sy = img.shape[0] / click.get("height", img.shape[0])
        pts.append((float(click["x"] * sx), float(click["y"] * sy)))
        st.session_state.points = pts
        st.rerun()
    return pts


def main():
    st.set_page_config(page_title="Telestration", page_icon="✏️", layout="wide")
    st.title("Telestration on endoscopy video")
    st.caption("Pause on a frame, draw a line A→B (or A→B→C), and let three trackers keep it on the "
               "tissue while the endoscope moves.")

    path, name = pick_video()
    if path is None:
        st.info("Put the endoscopy videos in the folder shown in the sidebar (or upload one).")
        return
    fps, n_total, w0, h0 = cached_info(path)
    if n_total <= 1:
        st.error("Could not read this video.")
        return

    clip_s = st.sidebar.slider("Clip length (s)", 2.0, 10.0, 5.0, 0.5)
    n_clip = int(round(clip_s * fps))
    max_start = max(0, n_total - n_clip - 1)
    # A long video has thousands of frames, so the slider alone is hard to aim with. The number
    # field jumps the slider to a time; the slider then fine-tunes frame by frame.
    slider_key = f"start_{path}_{n_clip}"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = min(int(fps * 5), max_start)

    def jump():
        st.session_state[slider_key] = min(int(st.session_state.jump_s * fps), max_start)

    st.sidebar.number_input("Jump to (s)", 0.0, max_start / fps, step=1.0, key="jump_s",
                            on_change=jump)
    start = st.sidebar.slider("Start frame", 0, max_start, key=slider_key,
                              help="The frame you pause on and draw on.")
    st.sidebar.caption(f"{name}: {w0}×{h0}, {fps:.1f} fps, {n_total / fps:.0f} s. "
                       f"Start at {start / fps:.1f} s, processed at width {PROC_WIDTH}.")
    n_points = 3 if st.sidebar.radio("Drawing", ["Line (2 points)", "Polyline (3 points)"],
                                     horizontal=True).startswith("Poly") else 2

    st.sidebar.header("2 · Preprocessing")
    use_clahe = st.sidebar.checkbox("CLAHE", True)
    clip_limit = st.sidebar.slider("CLAHE clipLimit", 1.0, 6.0, 2.0, 0.5, disabled=not use_clahe)
    tile = st.sidebar.slider("CLAHE tileGridSize", 2, 16, 8, 1, disabled=not use_clahe)
    use_glare = st.sidebar.checkbox("Specular-highlight mask", True)
    glare_thresh = st.sidebar.slider("Glare threshold", 180, 254, 220, 1, disabled=not use_glare)
    glare_dilate = st.sidebar.slider("Mask dilation (px)", 0, 20, 5, 1, disabled=not use_glare)
    use_border = st.sidebar.checkbox("Mask black border / overlays", True)
    prep = Prep(use_clahe, clip_limit, tile, use_glare, glare_thresh, glare_dilate, use_border)
    st.sidebar.header("3 · Tracking")
    fb_thresh = st.sidebar.slider(
        "LK forward-backward limit (px)", 0.5, 10.0, 4.0, 0.5,
        help="A Lucas-Kanade point is declared lost when tracking it forward and back again "
             "misses its start by more than this.")

    clip = cached_clip(path, start, n_clip)
    if len(clip) < 2:
        st.error("Could not read frames at this position.")
        return
    first = clip[0]

    # --- Step 1: drawing + preprocessing preview -------------------------------------------
    col_draw, col_prev = st.columns([3, 2], gap="large")
    with col_draw:
        st.subheader("Draw on the start frame")
        pts = click_points(first, n_points, f"{path}|{start}|{n_points}")
        c1, c2 = st.columns([1, 3])
        if c1.button("Reset points"):
            st.session_state.points = []
            st.rerun()
        c2.write(f"Points: {len(pts)}/{n_points}  " + "  ".join(
            f"{'ABC'[i]}=({x:.0f}, {y:.0f})" for i, (x, y) in enumerate(pts)))
    with col_prev:
        st.subheader("Preprocessing preview")
        before, after, n_before, n_after, pct = preview_panel(first, prep)
        st.image(before, caption=f"Before: raw grayscale, {n_before} FAST corners")
        st.image(after, caption=f"After: preprocessed, {n_after} corners kept, "
                                f"{pct:.1f}% masked (red = glare/border)")

    if len(pts) < n_points:
        st.info(f"Click {n_points - len(pts)} more point(s) on the image.")
        return

    run_key = (path, start, n_clip, tuple(pts), prep, fb_thresh)
    if st.button("▶ Track drawing", type="primary"):
        with st.spinner("Tracking (preprocessed run and baseline run)…"):
            tracks, lk_lost, df = run_trackers(clip, pts, prep, fb_thresh)
            _, _, df_base = run_trackers(clip, pts, NO_PREP, fb_thresh)
        st.session_state.result = dict(key=run_key, tracks=tracks, lk_lost=lk_lost, df=df,
                                       df_base=df_base)
        st.session_state.pop("video_cache", None)
    res = st.session_state.get("result")
    if res is None or res["key"] != run_key:
        st.info("Press **Track drawing** to run the trackers on this clip.")
        return
    tracks, lk_lost, df, df_base = res["tracks"], res["lk_lost"], res["df"], res["df_base"]

    # --- Step 2: output video + scrubbing ---------------------------------------------------
    st.divider()
    s = summary_numbers(tracks, lk_lost, df, df_base)
    m = st.columns(4)
    m[0].metric("LK points kept", f"{s['lk_kept']}/{s['n_pts']}",
                f"{s['lk_kept'] - s['lk_kept_base']:+d} vs. no preprocessing")
    m[1].metric("ORB valid homographies", f"{s['orb_ok'] * 100:.0f} %")
    m[2].metric("ORB mean inlier ratio", f"{s['orb_inl']:.2f}",
                f"{s['orb_inl'] - s['orb_inl_base']:+.2f} vs. no preprocessing")
    m[3].metric("Camera motion (ORB)", f"{s['motion']:.0f} px")

    show = st.multiselect("Trackers shown in the video", TRACKERS, default=TRACKERS,
                          format_func=lambda k: TRACKER_LABELS[k])
    show = [k for k in TRACKERS if k in show]
    col_vid, col_scrub = st.columns(2, gap="large")
    with col_vid:
        st.subheader("Annotated video")
        vkey = (run_key, tuple(show))
        cache = st.session_state.get("video_cache")
        if cache is None or cache[0] != vkey:
            with st.spinner("Encoding H.264…"):
                rendered = [render_frame(clip[t], t, fps, tracks, lk_lost, show)
                            for t in range(len(clip))]
                cache = (vkey, encode_h264(rendered, fps))
                st.session_state.video_cache = cache
        if cache[1] is not None:
            st.video(cache[1], format="video/mp4", loop=True)
        else:
            st.warning("H.264 encoding is not available (install imageio-ffmpeg). "
                       "Use the frame slider instead.")
    with col_scrub:
        st.subheader("Frame by frame")
        t = st.slider("Frame", 0, len(clip) - 1, len(clip) - 1)
        st.image(render_frame(clip[t], t, fps, tracks, lk_lost, show))

    # --- Step 3: statistics -----------------------------------------------------------------
    st.divider()
    st.subheader("Statistics")
    g1, g2 = st.columns(2)
    g1.plotly_chart(plot_displacement(tracks, fps), width="stretch")
    g2.plotly_chart(plot_features(df, fps), width="stretch")
    g3, g4 = st.columns(2)
    g3.plotly_chart(plot_lk(df, df_base, fps), width="stretch")
    g4.plotly_chart(plot_orb(df, df_base, fps), width="stretch")
    g5, _ = st.columns(2)
    g5.plotly_chart(plot_timing(df, fps), width="stretch")

    # --- Step 4: discussion -----------------------------------------------------------------
    st.divider()
    st.subheader("Discussion")
    st.markdown(discussion_markdown(s))


if __name__ == "__main__":
    main()
