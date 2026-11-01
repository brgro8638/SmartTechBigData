"""Own implementations (NumPy only), following SPEC.md step by step.

Every function mirrors the function with the same name in js/mia-core.js.
NumPy is used for vectorized array arithmetic, but the algorithms
themselves (filters, PCA, K-Means, Otsu, region growing, watershed) are
written here, not taken from a library.
"""

import heapq
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def percentile_sorted(sorted_values, p):
    """numpy.percentile(method="linear"), written out so JS can copy it exactly."""
    n = len(sorted_values)
    index = (p / 100) * (n - 1)
    lo = int(np.floor(index))
    hi = min(lo + 1, n - 1)
    t = index - lo
    a, b = float(sorted_values[lo]), float(sorted_values[hi])
    diff = b - a
    return b - diff * (1 - t) if t >= 0.5 else a + diff * t


def normalize(img, p_low, p_high):
    sorted_values = np.sort(img, axis=None)
    lo = percentile_sorted(sorted_values, p_low)
    hi = percentile_sorted(sorted_values, p_high)
    if hi <= lo:
        return np.zeros_like(img), lo, hi
    return np.clip((img - lo) / (hi - lo), 0, 1), lo, hi


def gaussian_kernel(sigma):
    r = int(np.floor(3 * sigma + 0.5))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp((-0.5 / (sigma * sigma)) * (x * x))
    return k / k.sum()


def correlate_axis(img, weights, axis):
    """1D correlation along one axis with edge mode "nearest" (clamped)."""
    weights = np.asarray(weights, dtype=np.float64)
    r = (len(weights) - 1) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    padded = np.pad(img, pad, mode="edge")
    out = np.zeros_like(img, dtype=np.float64)
    length = img.shape[axis]
    for j, w in enumerate(weights):
        # Window shifted by j - r relative to the output pixel
        out += w * (padded[j:j + length, :] if axis == 0 else padded[:, j:j + length])
    return out


def gaussian_blur(img, sigma):
    if sigma <= 0:
        return img.copy()
    k = gaussian_kernel(sigma)
    return correlate_axis(correlate_axis(img, k, 0), k, 1)


def sobel_magnitude(img):
    d, s = [-1, 0, 1], [1, 2, 1]
    gy = correlate_axis(correlate_axis(img, d, 0), s, 1)
    gx = correlate_axis(correlate_axis(img, d, 1), s, 0)
    return np.sqrt(gx * gx + gy * gy)


def local_std3(img):
    box = [1 / 3, 1 / 3, 1 / 3]
    mean = correlate_axis(correlate_axis(img, box, 0), box, 1)
    mean2 = correlate_axis(correlate_axis(img * img, box, 0), box, 1)
    return np.sqrt(np.maximum(mean2 - mean * mean, 0))


def min_max_scale(img):
    lo, hi = img.min(), img.max()
    if hi <= lo:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["intensity (smoothed)", "Gaussian σ=2", "Gaussian σ=4", "gradient |Sobel|", "local std 3×3"]


def build_features(norm, smooth):
    """Per-pixel feature stack, shape (n_pixels, 5)."""
    feats = [smooth, gaussian_blur(norm, 2), gaussian_blur(norm, 4), sobel_magnitude(smooth), local_std3(smooth)]
    return np.stack([f.ravel() for f in feats], axis=1)


def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    return (X - mean) / std


def pca(X, n_components):
    """Eigen-decomposition of the covariance matrix of the standardized features."""
    Z = standardize(X)
    cov = (Z.T @ Z) / (len(Z) - 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    components = vectors[:, :n_components].T.copy()
    for c in components:
        # Eigenvectors are only defined up to sign: make the largest loading positive
        if c[np.argmax(np.abs(c))] < 0:
            c *= -1
    explained = np.maximum(values[:n_components], 0) / np.maximum(values, 0).sum()
    return Z @ components.T, components, explained


# ---------------------------------------------------------------------------
# Clustering: K-Means with deterministic quantile initialisation
# ---------------------------------------------------------------------------

def quantile_init(X, k):
    order = np.argsort(X[:, 0], kind="stable")
    n = len(X)
    return np.array([X[order[int(np.floor((c + 0.5) * n / k))]] for c in range(k)], dtype=np.float64)


def _assign(X, centers):
    dist = np.zeros((len(centers), len(X)))
    for c, center in enumerate(centers):
        for j in range(X.shape[1]):
            diff = X[:, j] - center[j]
            dist[c] += diff * diff
    return np.argmin(dist, axis=0)  # first minimum = lowest cluster index on ties


def kmeans(X, k, max_iter, tol):
    centers = quantile_init(X, k)
    iterations = 0
    for it in range(max_iter):
        iterations = it + 1
        labels = _assign(X, centers)
        counts = np.bincount(labels, minlength=k)
        new_centers = centers.copy()
        for j in range(X.shape[1]):
            sums = np.bincount(labels, weights=X[:, j], minlength=k)
            nonempty = counts > 0  # empty clusters keep their centre
            new_centers[nonempty, j] = sums[nonempty] / counts[nonempty]
        shift = ((new_centers - centers) ** 2).sum(axis=1).max()
        centers = new_centers
        if shift <= tol:
            break
    labels = _assign(X, centers)
    # Relabel so that label 0 is the cluster with the lowest feature-0 centre
    rank = np.argsort(centers[:, 0], kind="stable")
    new_label = np.empty(k, dtype=np.int32)
    new_label[rank] = np.arange(k)
    return new_label[labels].astype(np.int32), centers[rank], iterations


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def otsu_threshold(img):
    bins = np.minimum(np.floor(img * 256), 255).astype(np.int64)
    hist = np.bincount(bins.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    mu_t = (np.arange(256) * hist).sum()
    w0 = np.cumsum(hist)[:255]
    mu = np.cumsum(np.arange(256) * hist)[:255]
    w1 = total - w0
    valid = (w0 > 0) & (w1 > 0)
    num = mu_t * w0 - mu * total
    sb = np.zeros(255)
    sb[valid] = (num[valid] * num[valid]) / (w0[valid] * w1[valid])
    return (int(np.argmax(sb)) + 1) / 256


def threshold(img, T):
    return (img >= T).astype(np.int32)


def _neighbours(p, H, W):
    y, x = divmod(p, W)
    if y > 0:
        yield p - W
    if y < H - 1:
        yield p + W
    if x > 0:
        yield p - 1
    if x < W - 1:
        yield p + 1


def region_grow(img, seed_y, seed_x, tol):
    """4-connected flood from the seed; criterion |value - seed value| <= tol."""
    H, W = img.shape
    flat = img.ravel()
    # Precomputing the criterion for all pixels keeps the Python loop small
    similar = (np.abs(flat - flat[seed_y * W + seed_x]) <= tol).tolist()
    out = np.zeros(H * W, dtype=np.int32)
    seed = seed_y * W + seed_x
    visited = bytearray(H * W)
    visited[seed] = 1
    stack = [seed]
    while stack:
        p = stack.pop()
        out[p] = 1
        for q in _neighbours(p, H, W):
            if not visited[q] and similar[q]:
                visited[q] = 1
                stack.append(q)
    return out.reshape(H, W)


def connected_components(mask):
    """4-connected labelling, labels 1..count in raster order. Returns (labels, sizes)."""
    H, W = mask.shape
    flat = mask.ravel().tolist()
    labels = [0] * (H * W)
    sizes = [0]
    count = 0
    for start in range(H * W):
        if not flat[start] or labels[start]:
            continue
        count += 1
        labels[start] = count
        queue = deque([start])
        size = 0
        while queue:
            p = queue.pop()
            size += 1
            for q in _neighbours(p, H, W):
                if flat[q] and not labels[q]:
                    labels[q] = count
                    queue.append(q)
        sizes.append(size)
    return np.array(labels, dtype=np.int32).reshape(H, W), np.array(sizes)


def watershed_markers(img, low, high, min_size):
    markers = np.zeros(img.shape, dtype=np.int32)
    markers[img < low] = 1
    components, sizes = connected_components(img > high)
    new_id = np.zeros(len(sizes), dtype=np.int32)
    keep = np.flatnonzero(sizes >= min_size)
    keep = keep[keep > 0]
    new_id[keep] = np.arange(2, 2 + len(keep))
    objects = new_id[components]
    markers[objects > 0] = objects[objects > 0]
    return markers, len(keep)


def watershed(elevation, markers):
    """Priority flood (Meyer): pixels are flooded in order of (elevation, insertion age)."""
    H, W = markers.shape
    elev = elevation.ravel().tolist()
    labels = markers.ravel().tolist()
    heap = []
    age = 0
    for p, lab in enumerate(labels):
        if lab > 0:
            heap.append((elev[p], age, p))
            age += 1
    heapq.heapify(heap)
    while heap:
        _, _, p = heapq.heappop(heap)
        for q in _neighbours(p, H, W):
            if labels[q] == 0:
                labels[q] = labels[p]
                heapq.heappush(heap, (elev[q], age, q))
                age += 1
    return np.maximum(np.array(labels, dtype=np.int32) - 1, 0).reshape(H, W)
