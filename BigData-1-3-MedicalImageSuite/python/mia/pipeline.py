"""Pipeline: one 2D slice + shared config -> labels, intermediate images, timings.

Mirrors runPipeline() in js/mia-core.js. `impl` selects the own NumPy
implementation or the library one; the steps and the config are the same.
"""

import copy
import time

import numpy as np

from . import library, own

IMPLEMENTATIONS = {"Python (own NumPy)": own, "Python (libraries)": library}
METHODS = {
    "kmeans": "K-Means clustering",
    "threshold": "Thresholding",
    "region_growing": "Region growing",
    "watershed": "Watershed",
}

DEFAULT_CONFIG = {
    "slice": 5,
    "normalize": {"p_low": 0.5, "p_high": 99.5},
    "smooth_sigma": 1.0,
    "pca": {"enabled": False, "n_components": 2},
    "method": "kmeans",
    "kmeans": {"k": 4, "max_iter": 100, "tol": 1e-6},
    "threshold": {"mode": "otsu", "value": 0.5},
    "region_growing": {"seed_x": 128, "seed_y": 128, "tolerance": 0.12},
    "watershed": {"low": 0.15, "high": 0.6, "min_marker_size": 20},
}


def default_config():
    return copy.deepcopy(DEFAULT_CONFIG)


def run_pipeline(img, config, impl=own):
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape
    timings, info = {}, {}
    start = t0 = time.perf_counter()

    def lap(name):
        nonlocal t0
        t = time.perf_counter()
        timings[name] = (t - t0) * 1000
        t0 = t

    norm, lo, hi = impl.normalize(img, config["normalize"]["p_low"], config["normalize"]["p_high"])
    info["normalize_range"] = [float(lo), float(hi)]
    lap("normalize")

    smooth = impl.gaussian_blur(norm, config["smooth_sigma"])
    lap("smooth")

    work, scores, components, pca_images = smooth, None, None, []
    if config["pca"]["enabled"]:
        X = impl.build_features(norm, smooth)
        scores, components, explained = impl.pca(X, config["pca"]["n_components"])
        pca_images = [impl.min_max_scale(scores[:, c].reshape(H, W)) for c in range(scores.shape[1])]
        work = pca_images[0]
        info["explained_variance"] = [float(e) for e in explained]
    lap("pca")

    method = config["method"]
    if method == "kmeans":
        X = scores if scores is not None else work.reshape(-1, 1)
        p = config["kmeans"]
        labels, _, iterations = impl.kmeans(X, p["k"], p["max_iter"], p["tol"])
        labels = labels.reshape(H, W)
        info["iterations"] = iterations
    elif method == "threshold":
        p = config["threshold"]
        T = impl.otsu_threshold(work) if p["mode"] == "otsu" else p["value"]
        labels = impl.threshold(work, T)
        info["threshold"] = float(T)
    elif method == "region_growing":
        p = config["region_growing"]
        sy = int(np.clip(p["seed_y"], 0, H - 1))
        sx = int(np.clip(p["seed_x"], 0, W - 1))
        labels = impl.region_grow(work, sy, sx, p["tolerance"])
        info["seed_value"] = float(work[sy, sx])
    elif method == "watershed":
        p = config["watershed"]
        elevation = impl.sobel_magnitude(work)
        markers, objects = impl.watershed_markers(work, p["low"], p["high"], p["min_marker_size"])
        labels = impl.watershed(elevation, markers)
        info["object_markers"] = int(objects)
    else:
        raise ValueError(f"Unknown method {method}")
    lap("method")
    timings["total"] = (time.perf_counter() - start) * 1000

    return {
        "labels": labels.astype(np.int32),
        "work": work,
        "pca_images": pca_images,
        "components": components,
        "timings": timings,
        "info": info,
    }
