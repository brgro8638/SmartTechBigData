"""Clustering module: preprocessing, six algorithms and quality metrics."""

import time
import warnings

import numpy as np
from sklearn.cluster import DBSCAN, Birch, KMeans, MeanShift, AgglomerativeClustering, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

ALGORITHMS = ["K-Means", "DBSCAN", "MeanShift", "BIRCH", "Agglomerative (Ward)", "Gaussian Mixture"]


def preprocess(features, use_pca, n_components):
    """Standardize, then optionally project with PCA.

    Standardizing is required because the features have different units
    (mm vs HU, pixels vs frames); without it the largest unit dominates.
    Returns (X, explained_variance_ratio or None).
    """
    X = StandardScaler().fit_transform(features)
    if use_pca:
        pca = PCA(n_components=n_components, random_state=0)
        X = pca.fit_transform(X)
        return X, pca.explained_variance_ratio_
    return X, None


def _fit_predict(name, X, params, seed):
    if name == "K-Means":
        model = KMeans(n_clusters=params["k"], n_init=10, random_state=seed)
        return model.fit_predict(X)

    if name == "DBSCAN":
        model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
        return model.fit_predict(X)

    if name == "MeanShift":
        # Bandwidth is estimated from a sample; quantile controls how wide the kernel is
        bandwidth = estimate_bandwidth(X, quantile=params["quantile"],
                                       n_samples=min(len(X), 2000), random_state=seed)
        # bin_seeding starts from a grid instead of every point, which makes it usable on 10k+ points
        model = MeanShift(bandwidth=max(bandwidth, 1e-3), bin_seeding=True)
        return model.fit_predict(X)

    if name == "BIRCH":
        model = Birch(threshold=params["threshold"], n_clusters=params["n_clusters"])
        return model.fit_predict(X)

    if name == "Agglomerative (Ward)":
        # Plain Ward needs an n x n distance matrix (~3 GB for 20k points).
        # A k-nearest-neighbour connectivity graph keeps it sparse and fast,
        # and it suits spatial data because only neighbouring points may merge.
        connectivity = kneighbors_graph(X, n_neighbors=params["n_neighbors"], include_self=False)
        model = AgglomerativeClustering(n_clusters=params["n_clusters"], linkage="ward",
                                        connectivity=connectivity)
        with warnings.catch_warnings():
            # Separate objects give a disconnected graph; sklearn links the parts itself and warns
            warnings.filterwarnings("ignore", message="the number of connected components")
            return model.fit_predict(X)

    if name == "Gaussian Mixture":
        model = GaussianMixture(n_components=params["n_components"], covariance_type="full",
                                random_state=seed)
        return model.fit(X).predict(X)

    raise ValueError(f"Unknown algorithm: {name}")


def run_algorithm(name, X, params, seed=42):
    """Cluster X and return (labels, runtime_seconds)."""
    start = time.perf_counter()
    labels = _fit_predict(name, X, params, seed)
    return labels, time.perf_counter() - start


def compute_metrics(X, labels, seed=42, silhouette_sample=5000):
    """Internal quality metrics (no ground truth exists for this data).

    Noise points (DBSCAN label -1) are excluded, otherwise DBSCAN would be
    punished for the points it deliberately refused to assign.
    Silhouette is O(n^2), so it is computed on a random sample.
    """
    labels = np.asarray(labels)
    noise_mask = labels == -1
    clusters = set(labels[~noise_mask])
    result = {
        "clusters": len(clusters),
        "noise_fraction": float(noise_mask.mean()),
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }
    X_valid, labels_valid = X[~noise_mask], labels[~noise_mask]
    # Metrics are only defined for 2 <= clusters < number of points
    if 2 <= len(clusters) < len(X_valid):
        sample = min(silhouette_sample, len(X_valid))
        result["silhouette"] = float(silhouette_score(X_valid, labels_valid,
                                                      sample_size=sample, random_state=seed))
        result["davies_bouldin"] = float(davies_bouldin_score(X_valid, labels_valid))
        result["calinski_harabasz"] = float(calinski_harabasz_score(X_valid, labels_valid))
    return result
