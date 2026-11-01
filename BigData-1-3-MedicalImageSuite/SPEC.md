# Shared algorithm specification

Both implementations (Python in `python/mia/own.py`, JavaScript in
`js/mia-core.js`) follow this document step by step. It is the contract that
makes the two languages produce comparable results. The Python *library*
implementation (`python/mia/library.py`) does the same steps with
scikit-learn / scikit-image / SciPy and is used as a reference.

## Data

- **Volume:** `.npy` file, shape `(slices, rows, cols)` or `(rows, cols)`,
  any numeric dtype. All processing is done on one 2D slice in float64.
- **Config:** JSON (see `config.example.json`). Both GUIs import and export
  the same file, so the same parameters can be run in both languages.
- **Result:** `labels.npy` (int32, `rows × cols`) + `result.json`
  (config, timings in ms, method info).

Label convention: `0` = background / not selected. K-Means labels are
`0 … k-1`, sorted by cluster centre (darkest cluster = 0).

## Synthetic phantom

Deterministic test volume (256×256×10) with ground truth, generated
identically in both languages with integer LCG noise
(`state = (state·1664525 + 1013904223) mod 2³²`, seed 12345) and only
basic arithmetic, so the two generators are bit-identical. Ground truth:
0 background, 1 skull, 2 brain, 3 ventricles, 4 lesions.

## Pipeline

1. **Normalize:** `lo, hi` = percentiles `p_low, p_high` (NumPy's linear
   interpolation), `norm = clip((img − lo)/(hi − lo), 0, 1)`.
2. **Smooth:** separable Gaussian, radius `floor(3σ + 0.5)`, edge mode
   *nearest* (clamp). Skipped when `σ = 0`.
3. **PCA (optional):** per-pixel features
   `[smooth, G₂(norm), G₄(norm), |Sobel(smooth)|, std₃ₓ₃(smooth)]`,
   standardized (z-score, population std). Covariance (n−1), eigenvectors
   sorted by eigenvalue, sign chosen so the largest-magnitude loading is
   positive. The *work image* is PC1 min-max scaled to `[0, 1]`.
   Without PCA the work image is the smoothed image.
4. **Method** (one of):
   - **K-Means:** features = first `n_components` PCA scores, or the work
     image alone without PCA. Init: sort points by feature 0 (stable),
     centre `c` = point at index `floor((c + 0.5)·n/k)`. Lloyd iterations
     until max squared centre shift ≤ `tol` or `max_iter`; empty clusters
     keep their centre; ties go to the lowest cluster index; final
     assignment with the final centres; relabel by centre feature 0.
   - **Threshold:** `label = work ≥ T`. Manual `T`, or Otsu: 256-bin
     histogram with bin `min(floor(v·256), 255)`, first `t` maximizing
     between-class variance, `T = (t + 1)/256`.
   - **Region growing:** 4-connected flood from the seed pixel, a pixel
     joins when `|work − work[seed]| ≤ tolerance`.
   - **Watershed:** elevation = `|Sobel(work)|`. Markers: background
     (label 1) where `work < low`; objects = 4-connected components of
     `work > high` with at least `min_marker_size` pixels (labels 2, 3, …
     in raster order). Priority flood with a min-heap ordered by
     (elevation, insertion order), neighbours up/down/left/right.
     Output label = marker label − 1.

## Comparison metrics

- **Matched pixel agreement:** labels of B are mapped to labels of A by
  maximum overlap (Hungarian in Python, greedy in JavaScript), then the
  fraction of equal pixels.
- **Adjusted Rand Index (ARI):** label-permutation invariant agreement
  (1 = identical partition).
- Against the phantom ground truth the same two metrics are used.
