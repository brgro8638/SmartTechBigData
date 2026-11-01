"""Library implementations of the same steps (SciPy, scikit-learn, scikit-image).

Used as a reference to check the own implementations against well-tested
code, and to compare runtime (compiled C/Cython vs. own NumPy / JavaScript).
Function names and return values match own.py so the pipeline can switch.
"""

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.segmentation import flood, watershed as sk_watershed
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import own

FEATURE_NAMES = own.FEATURE_NAMES


def normalize(img, p_low, p_high):
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(img), lo, hi
    return np.clip((img - lo) / (hi - lo), 0, 1), lo, hi


def gaussian_blur(img, sigma):
    if sigma <= 0:
        return img.copy()
    return ndimage.gaussian_filter(img, sigma, mode="nearest", truncate=3.0)


def sobel_magnitude(img):
    return np.hypot(ndimage.sobel(img, axis=1, mode="nearest"), ndimage.sobel(img, axis=0, mode="nearest"))


def local_std3(img):
    mean = ndimage.uniform_filter(img, 3, mode="nearest")
    mean2 = ndimage.uniform_filter(img * img, 3, mode="nearest")
    return np.sqrt(np.maximum(mean2 - mean * mean, 0))


min_max_scale = own.min_max_scale


def build_features(norm, smooth):
    feats = [smooth, gaussian_blur(norm, 2), gaussian_blur(norm, 4), sobel_magnitude(smooth), local_std3(smooth)]
    return np.stack([f.ravel() for f in feats], axis=1)


def pca(X, n_components):
    Z = StandardScaler().fit_transform(X)
    model = PCA(n_components=n_components, svd_solver="full").fit(Z)
    components = model.components_.copy()
    # Same sign convention as the own implementations (largest loading positive)
    for c in components:
        if c[np.argmax(np.abs(c))] < 0:
            c *= -1
    return Z @ components.T, components, model.explained_variance_ratio_


def kmeans(X, k, max_iter, tol):
    # Same deterministic start as the own implementation, so differences come
    # from the algorithm and not from a random initialisation
    model = KMeans(n_clusters=k, init=own.quantile_init(X, k), n_init=1,
                   max_iter=max_iter, tol=tol, algorithm="lloyd").fit(X)
    rank = np.argsort(model.cluster_centers_[:, 0], kind="stable")
    new_label = np.empty(k, dtype=np.int32)
    new_label[rank] = np.arange(k)
    return new_label[model.labels_].astype(np.int32), model.cluster_centers_[rank], int(model.n_iter_)


def otsu_threshold(img):
    return float(threshold_otsu(img, nbins=256))


def threshold(img, T):
    return (img >= T).astype(np.int32)


def region_grow(img, seed_y, seed_x, tol):
    return flood(img, (seed_y, seed_x), connectivity=1, tolerance=tol).astype(np.int32)


def watershed_markers(img, low, high, min_size):
    markers = np.zeros(img.shape, dtype=np.int32)
    markers[img < low] = 1
    components, count = ndimage.label(img > high)  # default structure = 4-connectivity
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    keep = np.flatnonzero(sizes >= min_size)
    keep = keep[keep > 0]
    new_id = np.zeros(count + 1, dtype=np.int32)
    new_id[keep] = np.arange(2, 2 + len(keep))
    objects = new_id[components]
    markers[objects > 0] = objects[objects > 0]
    return markers, len(keep)


def watershed(elevation, markers):
    labels = sk_watershed(elevation, markers, connectivity=1)
    return np.maximum(labels.astype(np.int32) - 1, 0)
