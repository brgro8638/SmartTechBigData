"""
Automatic box pose estimation from a single frame, no clicks and no reference needed.

Analysis-by-synthesis:
  1. Segment the box (binary mask of the box in the frame).
  2. Coarse search: sample many orientations; for each, place the model so its projected
     silhouette has the mask's area and centroid, and score the overlap (IoU).
  3. Refine the best candidates with a local search that also rewards the model's visible
     edges lying on long straight image edges (this removes the front/back flip ambiguity).
"""

import cv2
import numpy as np

# Rotations that map a cuboid onto itself (visually identical poses).
_SYMS = [np.diag(d) for d in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]

# (corner indices, outward normal) for each face; corner order matches the app's corners_3d.
_FACES = [
    ((0, 1, 2, 3), (0, 0, -1)),
    ((4, 5, 6, 7), (0, 0, 1)),
    ((0, 1, 5, 4), (0, -1, 0)),
    ((2, 3, 7, 6), (0, 1, 0)),
    ((1, 2, 6, 5), (1, 0, 0)),
    ((0, 3, 7, 4), (-1, 0, 0)),
]
_SIGNS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                   [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)

def _edge_faces():
    """Map each box edge to the indices of the two faces that share it."""
    m = {}
    for fi, (idx, _) in enumerate(_FACES):
        for k in range(4):
            e = tuple(sorted((idx[k], idx[(k + 1) % 4])))
            m.setdefault(e, []).append(fi)
    return m

_EDGE_FACES = _edge_faces()

#region Box Segmentation
def segment_box(frame_bgr, scale=0.5):
    """Binary mask (0/255, full resolution) of the box, or None.

    A strict colour seed (the box is much more saturated than the dark table), refined with
    GrabCut so the outline follows the real box edges.
    """
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    seed = cv2.inRange(hsv, (8, 95, 85), (100, 255, 255))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < 600:
        return None
    seed = np.zeros_like(seed)
    cv2.drawContours(seed, [big], -1, 255, -1)

    gc = np.full(seed.shape, cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(seed, np.ones((21, 21), np.uint8)) > 0] = cv2.GC_PR_BGD
    gc[seed > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(seed, np.ones((9, 9), np.uint8)) > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(small, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
        res = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    except cv2.error:
        res = seed
    cnts, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(res)
    cv2.drawContours(out, [best], -1, 255, -1)
    return cv2.resize(out, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

def _rodrigues(r):
    return cv2.Rodrigues(np.asarray(r, float).reshape(3, 1))[0]

def _pattern_search(f, x0, steps, min_scale=1 / 32, max_evals=400):
    """Maximise f by coordinate pattern search with step halving."""
    x = np.array(x0, float)
    fx = f(x)
    scale, evals = 1.0, 1
    while scale > min_scale and evals < max_evals:
        improved = False
        for i in range(len(x)):
            for sgn in (1.0, -1.0):
                y = x.copy()
                y[i] += sgn * steps[i] * scale
                fy = f(y)
                evals += 1
                if fy > fx:
                    x, fx, improved = y, fy, True
                    break
        if not improved:
            scale *= 0.5
    return x, fx

class _Target:
    """Mask + edge distance map prepared at one working scale."""

    def __init__(self, gray, mask, K, scale):
        self.scale = scale
        self.h, self.w = int(round(gray.shape[0] * scale)), int(round(gray.shape[1] * scale))
        self.mask = (cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        self.area = float(self.mask.sum())
        m = cv2.moments(self.mask, binaryImage=True)
        self.centroid = np.array([m["m10"] / max(m["m00"], 1), m["m01"] / max(m["m00"], 1)])
        S = np.diag([scale, scale, 1.0])
        self.K = S @ np.asarray(K, float)
        self.fx, self.fy, self.cx, self.cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        self.dist = None  # distance-to-long-edge map, built lazily

    def build_edges(self, gray, min_len_px=40):
        small = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(self.mask, np.ones((15, 15), np.uint8))
        canvas = np.zeros((self.h, self.w), np.uint8)
        min_len = min_len_px * self.scale
        lines = None
        try:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
            res = lsd.detect(small)[0]
            if res is not None:
                lines = res.reshape(-1, 4)
        except (AttributeError, cv2.error):
            edges = cv2.Canny(small, 60, 150)
            hl = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=int(min_len), maxLineGap=4)
            if hl is not None:
                lines = hl.reshape(-1, 4).astype(float)
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                if np.hypot(x2 - x1, y2 - y1) < min_len:
                    continue
                mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if 0 <= mx < self.w and 0 <= my < self.h and roi[my, mx]:
                    cv2.line(canvas, (int(round(x1)), int(round(y1))),
                             (int(round(x2)), int(round(y2))), 255, 1)
        self.dist = cv2.distanceTransform(255 - canvas, cv2.DIST_L2, 3).astype(np.float32)

#region Box Fitter
class BoxFitter:
    def __init__(self, bmin, bmax, K):
        bmin, bmax = np.asarray(bmin, float), np.asarray(bmax, float)
        self.c = (bmin + bmax) / 2.0
        self.h = (bmax - bmin) / 2.0
        self.pts = _SIGNS * self.h # corners relative to the box centre
        self.K = np.asarray(K, float)
        self.face_n = np.array([n for _, n in _FACES], float)
        self.face_c = self.face_n * self.h   
        self.edges = [(e, fs) for e, fs in _EDGE_FACES.items()]

    def _project(self, R, tc, T):
        pc = self.pts @ R.T + tc
        if (pc[:, 2] < 0.2).any():
            return None, None
        uv = np.column_stack([pc[:, 0] / pc[:, 2] * T.fx + T.cx, pc[:, 1] / pc[:, 2] * T.fy + T.cy])
        return uv, pc

    def _iou(self, uv, T):
        hull = cv2.convexHull(uv.astype(np.float32))
        img = np.zeros((T.h, T.w), np.uint8)
        cv2.fillConvexPoly(img, np.round(hull * 16).astype(np.int32), 1, cv2.LINE_8, 4)
        inter = float((img & T.mask).sum())
        union = float(img.sum()) + T.area - inter
        return inter / union if union > 0 else 0.0

    def _edge_score(self, R, tc, pc, T, sigma):
        vis_face = ((self.face_c @ R.T + tc) * (self.face_n @ R.T)).sum(1) < 0
        t = np.linspace(0.0, 1.0, 14)[:, None]
        total, count = 0.0, 0
        for (a, b), fs in self.edges:
            if not any(vis_face[f] for f in fs):
                continue
            p = pc[a] * (1 - t) + pc[b] * t
            x = np.round(p[:, 0] / p[:, 2] * T.fx + T.cx).astype(int)
            y = np.round(p[:, 1] / p[:, 2] * T.fy + T.cy).astype(int)
            ok = (x >= 0) & (x < T.w) & (y >= 0) & (y < T.h)
            d = np.full(len(t), 30.0 * T.scale)
            d[ok] = T.dist[y[ok], x[ok]]
            total += float(np.exp(-(d / sigma) ** 2).sum())
            count += len(t)
        return total / count if count else 0.0

    def score(self, R, tc, T, edge_weight):
        uv, pc = self._project(R, tc, T)
        if uv is None:
            return 0.0
        s = self._iou(uv, T)
        if edge_weight > 0 and T.dist is not None and s > 0.3:
            s += edge_weight * self._edge_score(R, tc, pc, T, sigma=3.0 * max(T.scale * 2, 1.0))
        return s

    def _place(self, R, T):
        """Translation putting the projected hull on the mask's area and centroid."""
        z_ref = 10.0
        uv, _ = self._project(R, np.array([0, 0, z_ref]), T)
        if uv is None:
            return None
        hull = cv2.convexHull(uv.astype(np.float32))
        area = max(cv2.contourArea(hull), 1e-3)
        z = z_ref * np.sqrt(area / max(T.area, 1.0))
        tc = np.array([0.0, 0.0, z])
        uv, _ = self._project(R, tc, T)
        if uv is None:
            return None
        m = cv2.moments(cv2.convexHull(uv.astype(np.float32)))
        hc = np.array([m["m10"] / max(m["m00"], 1e-6), m["m01"] / max(m["m00"], 1e-6)])
        d = T.centroid - hc
        tc[0] += d[0] * z / T.fx
        tc[1] += d[1] * z / T.fy
        return tc

    def _refine(self, R0, tc0, T, edge_weight, max_evals):
        z0 = tc0[2]

        def pose(x):
            return _rodrigues(x[:3]) @ R0, np.array(x[3:], float)

        def f(x):
            R, tc = pose(x)
            return self.score(R, tc, T, edge_weight)

        x0 = np.concatenate([np.zeros(3), tc0])
        steps = np.array([0.12, 0.12, 0.12, 0.05 * z0, 0.05 * z0, 0.08 * z0])
        x, fx = _pattern_search(f, x0, steps, max_evals=max_evals)
        R, tc = pose(x)
        return R, tc, fx

    def fit(self, frame_bgr, mask, n_samples=5000, n_coarse=30, n_fine=6, seed=0):
        """Return dict(rvec, tvec, iou, edge, score) in OpenCV convention, or None."""
        if mask is None or int((mask > 0).sum()) < 500:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        rng = np.random.default_rng(seed)

        # coarse orientation search on a quarter-resolution mask
        T4 = _Target(gray, mask, self.K, 0.25)
        q = rng.normal(size=(n_samples, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        cands = []
        for qi in q:
            w, x, y, z = qi
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            tc = self._place(R, T4)
            if tc is None:
                continue
            cands.append((self.score(R, tc, T4, 0.0), R, tc))
        cands.sort(key=lambda c: -c[0])

        picked = []
        for s, R, tc in cands:
            if all(min(np.arccos(np.clip((np.trace(R0.T @ R @ S) - 1) / 2, -1, 1)) for S in _SYMS) > 0.3
                   for _, R0, _ in picked):
                picked.append((s, R, tc))
            if len(picked) >= n_coarse:
                break

        # silhouette-only refinement of each candidate
        refined = []
        for s, R, tc in picked:
            R2, tc2, s2 = self._refine(R, tc, T4, 0.0, max_evals=150)
            refined.append((s2, R2, tc2))
        refined.sort(key=lambda c: -c[0])

        # silhouette + edge refinement of the best few at half resolution
        T2 = _Target(gray, mask, self.K, 0.5)
        T2.build_edges(gray)
        best = None
        for s, R, tc in refined[:n_fine]:
            R2, tc2, s2 = self._refine(R, tc, T2, 0.6, max_evals=300)
            if best is None or s2 > best[0]:
                best = (s2, R2, tc2)
        if best is None:
            return None

        # report quality at half resolution
        _, R, tc = best
        uv, pc = self._project(R, tc, T2)
        iou = self._iou(uv, T2) if uv is not None else 0.0
        edge = self._edge_score(R, tc, pc, T2, 3.0 * max(T2.scale * 2, 1.0)) if uv is not None else 0.0
        rvec = cv2.Rodrigues(R)[0]
        # model points are expressed relative to the OBJ origin in the app: X_cam = R (P - c) + tc
        tvec = (tc - R @ self.c).reshape(3, 1)
        return {"rvec": rvec, "tvec": tvec, "iou": iou, "edge": edge, "score": best[0]}
