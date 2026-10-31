# 2D→3D Clustering Visualizer (Big Data 1-6)

Streamlit application for issue
[#9](https://github.com/brgro8638/SmartTechBigData/issues/9). It clusters a
3D DICOM series (CT or MR) or the differential frames of a video with six
algorithms, and compares the results quantitatively (metrics table) and
visually (interactive 3D point clouds, one colour per cluster).

## Run

From the repository root:

```bash
python -m pip install -r BigData-1-6-Visualizer3D/requirements.txt
streamlit run BigData-1-6-Visualizer3D/app.py
```

Put `dicoms.zip` (from Canvas) or a video in `BigData-1-6-Visualizer3D/data/`
and it appears in the sidebar, or upload a file in the app. The `data/` folder
is ignored by git: DICOM files contain patient data and must not be committed.

## How it works

| Step | DICOM | Video |
|---|---|---|
| Points | Voxels inside the chosen intensity range, downsampled (stride) and randomly subsampled | Pixels where `|frame[t+1] − frame[t]|` exceeds a threshold |
| Features | x, y, z (mm), intensity (HU for CT, signal for MR) | x, y (px), t (frame), difference value |
| 3D axes | x, y, z in mm, real proportions | x, y, time |

Features are standardized (mean 0, std 1) before clustering because they have
different units. PCA can optionally be applied after standardization.

**Algorithms:** K-Means, DBSCAN, MeanShift, BIRCH, Agglomerative (Ward with a
k-nearest-neighbour connectivity graph), Gaussian Mixture. All parameters are
set in the sidebar.

**Metrics:** number of clusters, noise fraction, silhouette (on a sample),
Davies-Bouldin, Calinski-Harabasz and runtime. DBSCAN noise points are
excluded from the metrics.

The *Compare algorithms* tab runs the selected algorithms on the same points
and exports the table (CSV) and an HTML report with all 3D plots.

## Notes on the provided data

`dicoms.zip` is an MR examination (MR angiography of the neck), not a CT, and
holds 12 series. The app lets you pick the series. MR has no Hounsfield units,
so for non-CT data the intensity range follows the series' own values; the
default keeps the brightest 1% of voxels, which in the angiography are the
vessels.

## Files

- `app.py`: Streamlit GUI
- `loaders.py`: DICOM and video loading, conversion to point clouds
- `clustering.py`: preprocessing, the six algorithms and the metrics
