"""Synthetic brain-like phantom with ground truth.

Bit-identical to makePhantom() in js/mia-core.js: the noise comes from an
integer LCG and every pixel value uses only +, -, *, / in the same order, so
both languages produce exactly the same float32 volume. This gives a test
dataset with known ground truth that can be committed (no patient data).
"""

import numpy as np

# centre x, centre y, radius, slice range t_min..t_max
LESIONS = [
    (0.35, 0.30, 0.09, -0.6, 0.6),
    (-0.30, 0.40, 0.06, -0.3, 1.0),
    (0.0, -0.50, 0.05, -1.0, 1.0),
]
GROUND_TRUTH_NAMES = ["background", "skull", "brain", "ventricles", "lesions"]


def _lcg_noise(count, seed=12345):
    """(u1 + u2 - 1) * 40 for consecutive pairs of LCG draws, in raster order."""
    out = np.empty(count, dtype=np.float64)
    state = seed
    for i in range(count):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        u1 = state / 4294967296
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        u2 = state / 4294967296
        out[i] = (u1 + u2 - 1) * 40
    return out


def make_phantom(size=256, slices=10):
    """Return (volume float32 (Z, H, W), ground_truth uint8 (Z, H, W))."""
    H = W = size
    Z = slices
    noise = _lcg_noise(Z * H * W).reshape(Z, H, W)
    volume = np.empty((Z, H, W), dtype=np.float32)
    gt = np.zeros((Z, H, W), dtype=np.uint8)

    v = ((np.arange(H) - (H - 1) / 2) / (H / 2))[:, None] * np.ones((1, W))
    u = ((np.arange(W) - (W - 1) / 2) / (W / 2))[None, :] * np.ones((H, 1))
    half = (Z - 1) / 2

    for z in range(Z):
        t = (z - half) / half if Z > 1 else 0.0
        s = 1 - 0.35 * t * t
        value = np.zeros((H, W))
        label = np.zeros((H, W), dtype=np.uint8)

        ou, ov = u / (0.85 * s), v / (0.95 * s)
        head = ou * ou + ov * ov <= 1
        iu, iv = u / (0.78 * s), v / (0.88 * s)
        brain = head & (iu * iu + iv * iv <= 1)

        label[head] = 1
        value[head] = 900
        label[brain] = 2
        value[brain] = 400
        if abs(t) < 0.7:
            for cx in (-0.12, 0.12):
                du, dv = (u - cx) / (0.07 * s), (v + 0.05) / (0.22 * s)
                inside = brain & (du * du + dv * dv <= 1)
                label[inside] = 3
                value[inside] = 150
        for cx, cy, r, t_min, t_max in LESIONS:
            if t < t_min or t > t_max:
                continue
            du, dv = (u - cx) / r, (v - cy) / r
            inside = brain & (du * du + dv * dv <= 1)
            label[inside] = 4
            value[inside] = 650

        value = np.where(head, value * (1 + 0.15 * u), 0.0)
        volume[z] = (value + noise[z]).astype(np.float32)
        gt[z] = label
    return volume, gt
