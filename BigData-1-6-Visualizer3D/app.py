"""2D -> 3D Clustering Visualizer (Streamlit GUI).

Clusters a DICOM series (CT or MR) or the differential frames of a video with six
algorithms and compares them quantitatively (metrics table) and visually
(interactive 3D point clouds).

Run from this folder:   streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clustering import ALGORITHMS, compute_metrics, preprocess, run_algorithm
from loaders import (build_volume, diffs_to_points, difference_frames, load_video_frames,
                     read_dicom_series, volume_to_points)

DATA_DIR = Path(__file__).parent / "data"
PALETTE = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
NOISE_COLOR = "lightgray"
MAX_LEGEND_CLUSTERS = 40

st.set_page_config(page_title="2D→3D Clustering Visualizer", layout="wide")


# ---------------------------------------------------------------------------
# Cached loading. Arguments starting with "_" are not hashed by Streamlit,
# so a short key string identifies the file instead of hashing 100+ MB.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Reading DICOM files...", max_entries=2)
def cached_dicom_series(key, _read_bytes):
    # cache_resource (not cache_data) avoids copying the datasets and volumes on every rerun
    return read_dicom_series(_read_bytes())


@st.cache_resource(show_spinner="Building volume...", max_entries=4)
def cached_dicom_volume(key, _datasets):
    return build_volume(_datasets)


@st.cache_data(show_spinner="Extracting points from volume...", max_entries=10)
def cached_dicom_points(key, _volume, spacing, value_min, value_max, stride, max_points, seed, intensity_name):
    return volume_to_points(_volume, spacing, value_min, value_max, stride, max_points, seed, intensity_name)


@st.cache_data(show_spinner="Decoding video...", max_entries=4)
def cached_video_frames(key, _read_bytes, suffix, resize_width, frame_step, max_frames):
    return load_video_frames(_read_bytes(), suffix, resize_width, frame_step, max_frames)


@st.cache_data(show_spinner="Computing differential frames...", max_entries=10)
def cached_video_diffs(key, _frames, blur):
    return difference_frames(_frames, blur)


@st.cache_data(show_spinner="Extracting moving pixels...", max_entries=10)
def cached_video_points(key, _diffs, threshold, max_points, seed):
    return diffs_to_points(_diffs, threshold, max_points, seed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def choose_file(label, extensions, uploader_types):
    """Let the user pick a file from data/ or upload one. Returns (key, read_bytes, name) or None."""
    local = sorted(p for p in DATA_DIR.glob("*") if p.suffix.lower() in extensions) if DATA_DIR.exists() else []
    options = [p.name for p in local] + ["Upload a file..."]
    choice = st.sidebar.selectbox(label, options)
    if choice != "Upload a file...":
        path = DATA_DIR / choice
        key = f"{path}:{path.stat().st_mtime}"
        return key, path.read_bytes, path.name
    uploaded = st.sidebar.file_uploader("Upload", type=uploader_types)
    if uploaded is None:
        return None
    key = f"upload:{uploaded.name}:{uploaded.size}:{uploaded.file_id}"
    return key, uploaded.getvalue, uploaded.name


def to_uint8(image, low, high):
    """Linear window to 0-255 for display."""
    scaled = (np.clip(image, low, high) - low) / max(high - low, 1e-6)
    return (scaled * 255).astype(np.uint8)


def cluster_colors(labels):
    return [NOISE_COLOR if lab == -1 else PALETTE[lab % len(PALETTE)] for lab in labels]


def make_3d_figure(coords, axis_names, labels, title, aspectmode, height=600):
    """Interactive 3D point cloud, one distinct colour per cluster (noise in grey)."""
    labels = np.asarray(labels)
    unique = sorted(set(labels))
    fig = go.Figure()
    if len(unique) <= MAX_LEGEND_CLUSTERS:
        # One trace per cluster gives a legend where clusters can be toggled on/off
        for lab in unique:
            mask = labels == lab
            name = "noise" if lab == -1 else f"cluster {lab}"
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2], mode="markers",
                marker=dict(size=2, color=cluster_colors([lab])[0], opacity=0.8),
                name=f"{name} ({mask.sum()})"))
    else:
        # Hundreds of traces make the browser crawl; use one trace with per-point colours
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers",
            marker=dict(size=2, color=cluster_colors(labels), opacity=0.8),
            name=f"{len(unique)} clusters", text=[f"cluster {lab}" for lab in labels]))
    fig.update_layout(
        title=title, height=height, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
        scene=dict(xaxis_title=axis_names[0], yaxis_title=axis_names[1],
                   zaxis_title=axis_names[2], aspectmode=aspectmode))
    return fig


def format_params(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


# ---------------------------------------------------------------------------
# Sidebar: data source
# ---------------------------------------------------------------------------

st.sidebar.title("2D→3D Clustering")
st.sidebar.header("1. Data")
source = st.sidebar.radio("Data source", ["DICOM series (CT/MR)", "Video (differential frames)"])
seed = int(st.sidebar.number_input("Random seed", value=42, step=1,
                                   help="Fixed seed so subsampling and results are reproducible."))
max_points = st.sidebar.slider("Max points", 1000, 30000, 10000, 1000,
                               help="Random subsample. DBSCAN, MeanShift and Ward scale roughly O(n²).")

points = None
preview = None

if source == "DICOM series (CT/MR)":
    picked = choose_file("DICOM zip", {".zip"}, ["zip"])
    if picked is not None:
        key, read_bytes, name = picked
        try:
            series_list = cached_dicom_series(key, read_bytes)
        except Exception as error:
            st.error(f"Could not read DICOM files: {error}")
            st.stop()
        # An examination holds several series; only one can become a volume
        series_index = st.sidebar.selectbox(
            "Series", range(len(series_list)),
            format_func=lambda i: (f"{series_list[i]['modality']} · {series_list[i]['description']} · "
                                   f"{series_list[i]['images']} images · "
                                   f"{series_list[i]['shape'][0]}×{series_list[i]['shape'][1]}"))
        series = series_list[series_index]
        volume_key = f"{key}:{series['uid']}"
        volume, spacing, vol_info = cached_dicom_volume(volume_key, series["datasets"])

        is_ct = vol_info["modality"] == "CT"
        intensity_name = "HU" if is_ct else "signal"
        if is_ct:
            range_limits, range_default = (-1024, 3000), (-500, 3000)
            range_help = "Voxels outside are dropped. The default removes air."
        else:
            # MR signal has no fixed unit, so the slider follows this series' own values.
            # Keeping only the brightest ~1% isolates e.g. the vessels in an MR angiography.
            low, high = int(volume.min()), int(volume.max())
            range_limits, range_default = (low, high), (int(np.percentile(volume, 99)), high)
            range_help = ("MR signal has no unit like HU. The default keeps the brightest 1% of voxels "
                          "(vessels in an MR angiography); lower the minimum to include more tissue.")
        value_min, value_max = st.sidebar.slider(
            f"{intensity_name} range kept", range_limits[0], range_limits[1], range_default,
            key=f"range:{volume_key}", help=range_help)
        stride = st.sidebar.slider("Voxel stride", 1, 8, 2,
                                   help="Take every n-th voxel in each direction before sampling.")
        points = cached_dicom_points(volume_key, volume, spacing, value_min, value_max, stride,
                                     max_points, seed, intensity_name)
        axis_names, aspectmode = ["x (mm)", "y (mm)", "z (mm)"], "data"
        spatial_cols = ["x", "y", "z"]
        name = f"{name} – {series['description']}"
        preview = ("dicom", volume, spacing, vol_info, value_min, value_max, name)
else:
    picked = choose_file("Video", {".mp4", ".avi", ".mov", ".mkv"}, ["mp4", "avi", "mov", "mkv"])
    if picked is not None:
        key, read_bytes, name = picked
        resize_width = st.sidebar.slider("Resize width (px)", 64, 480, 160, 16,
                                         help="Smaller frames = fewer moving pixels = faster clustering.")
        frame_step = st.sidebar.slider("Use every n-th frame", 1, 10, 1)
        max_frames = st.sidebar.slider("Max frames", 10, 1000, 200, 10)
        blur = st.sidebar.select_slider("Blur kernel", [1, 3, 5, 7, 9], 5,
                                        help="Gaussian blur before differencing to suppress noise.")
        threshold = st.sidebar.slider("Difference threshold", 1, 255, 25,
                                      help="A pixel counts as moving if |frame[t+1]-frame[t]| exceeds this.")
        try:
            frames, fps = cached_video_frames(key, read_bytes, Path(name).suffix,
                                              resize_width, frame_step, max_frames)
        except Exception as error:
            st.error(f"Could not load video: {error}")
            st.stop()
        frame_key = f"{key}:{resize_width}:{frame_step}:{max_frames}"
        diffs = cached_video_diffs(frame_key, frames, blur)
        points = cached_video_points(f"{frame_key}:{blur}", diffs, threshold, max_points, seed)
        axis_names, intensity_name, aspectmode = ["x (px)", "y (px)", "t (frame)"], "diff", "cube"
        spatial_cols = ["x", "y", "t"]
        preview = ("video", frames, diffs, fps, threshold, name)

# ---------------------------------------------------------------------------
# Sidebar: features, PCA and algorithm parameters
# ---------------------------------------------------------------------------

st.sidebar.header("2. Features")
use_intensity = st.sidebar.checkbox("Use intensity as a feature", value=True,
                                    help="HU for DICOM, difference value for video. "
                                         "Off = cluster on position only.")
use_pca = st.sidebar.checkbox("Apply PCA", value=False)
n_features = 4 if use_intensity else 3
n_components = st.sidebar.slider("PCA components", 1, n_features, min(3, n_features), disabled=not use_pca)

st.sidebar.header("3. Algorithm parameters")
params = {}
with st.sidebar.expander("K-Means"):
    params["K-Means"] = {"k": st.slider("k (clusters)", 2, 20, 4, key="km_k")}
with st.sidebar.expander("DBSCAN"):
    params["DBSCAN"] = {
        "eps": st.slider("eps", 0.05, 2.0, 0.3, 0.05, key="db_eps",
                         help="Neighbourhood radius in standardized feature space."),
        "min_samples": st.slider("min_samples", 2, 100, 10, key="db_min"),
    }
with st.sidebar.expander("MeanShift"):
    params["MeanShift"] = {
        "quantile": st.slider("bandwidth quantile", 0.02, 0.5, 0.1, 0.01, key="ms_q",
                              help="Larger = wider kernel = fewer clusters."),
    }
with st.sidebar.expander("BIRCH"):
    params["BIRCH"] = {
        "threshold": st.slider("threshold", 0.1, 2.0, 0.5, 0.05, key="bi_t",
                               help="Max radius of a subcluster in the CF tree."),
        "n_clusters": st.slider("n_clusters", 2, 20, 4, key="bi_n"),
    }
with st.sidebar.expander("Agglomerative (Ward)"):
    params["Agglomerative (Ward)"] = {
        "n_clusters": st.slider("n_clusters", 2, 20, 4, key="ag_n"),
        "n_neighbors": st.slider("connectivity neighbours", 5, 50, 10, key="ag_nn",
                                 help="Only points among each other's k nearest neighbours can merge."),
    }
with st.sidebar.expander("Gaussian Mixture"):
    params["Gaussian Mixture"] = {"n_components": st.slider("n_components", 2, 20, 4, key="gm_n")}

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("2D→3D Clustering Visualizer")

if points is None:
    st.info("Choose a DICOM zip or a video in the sidebar. Files placed in the `data/` folder "
            "next to app.py are listed automatically.")
    st.stop()
if len(points) < 10:
    st.warning(f"Only {len(points)} points after filtering. Widen the HU range or lower the threshold.")
    st.stop()

feature_cols = spatial_cols + ([intensity_name] if use_intensity else [])
X, explained = preprocess(points[feature_cols].to_numpy(dtype=np.float64), use_pca, n_components)
# Everything that changes X; used to detect stale results after the user moves a slider
data_signature = (str(preview[-1]), len(points), float(points[spatial_cols[0]].sum()),
                  tuple(feature_cols), use_pca, n_components)

# Plot in real coordinates by default; PCA space is optional when there are >= 3 components
plot_space = "Original coordinates"
if use_pca and n_components >= 3:
    plot_space = st.radio("Plot in", ["Original coordinates", "PCA space (PC1-PC3)"], horizontal=True)
if plot_space == "Original coordinates":
    coords, plot_axes, plot_aspect = points[spatial_cols].to_numpy(), axis_names, aspectmode
else:
    coords, plot_axes, plot_aspect = X[:, :3], ["PC1", "PC2", "PC3"], "cube"

col1, col2, col3 = st.columns(3)
col1.metric("Points", f"{len(points):,}")
col2.metric("Features", ", ".join(feature_cols))
col3.metric("PCA variance kept", f"{explained.sum():.1%}" if explained is not None else "PCA off")

tab_data, tab_single, tab_compare = st.tabs(["Data", "Single algorithm", "Compare algorithms"])

# --- Data tab: shows what the point cloud is made from -----------------------
with tab_data:
    if preview[0] == "dicom":
        _, volume, spacing, vol_info, value_min, value_max, name = preview
        st.write(f"**{name}** ({vol_info['modality']}): {vol_info['slices']} slices, "
                 f"volume {volume.shape} (z, y, x), "
                 f"spacing {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} mm, "
                 f"{intensity_name} range {volume.min():.0f} to {volume.max():.0f}.")
        slice_index = st.slider("Slice", 0, volume.shape[0] - 1, volume.shape[0] // 2)
        c1, c2 = st.columns(2)
        image = volume[slice_index]
        if vol_info["modality"] == "CT":
            window, window_caption = (40 - 200, 40 + 200), "Soft-tissue window (W400/L40)"
        else:
            # A fixed HU window is meaningless for MR; stretch the series' own 1st-99.5th percentile instead
            window = tuple(np.percentile(volume, [1, 99.5]))
            window_caption = "Auto window (1st–99.5th percentile of the series)"
        c1.image(to_uint8(image, *window), caption=window_caption, width="stretch")
        kept = ((image >= value_min) & (image <= value_max)).astype(np.uint8) * 255
        c2.image(kept, caption=f"Voxels kept by {intensity_name} range [{value_min}, {value_max}]",
                 width="stretch")
    else:
        _, frames, diffs, fps, threshold, name = preview
        st.write(f"**{name}**: {len(frames)} frames used ({fps:.1f} fps source), "
                 f"frame size {frames[0].shape[1]} × {frames[0].shape[0]} px.")
        # Start at the frame with the most motion so the preview is informative
        busiest = int(np.argmax([d.sum() for d in diffs]))
        t = st.slider("Differential frame t", 0, len(diffs) - 1, busiest)
        c1, c2, c3 = st.columns(3)
        c1.image(frames[t], caption=f"Frame {t}", width="stretch")
        c2.image(to_uint8(diffs[t], 0, max(int(diffs[t].max()), 1)),
                 caption="|frame[t+1] − frame[t]|", width="stretch")
        c3.image((diffs[t] > threshold).astype(np.uint8) * 255,
                 caption=f"Moving pixels (> {threshold})", width="stretch")
    st.subheader("Point cloud before clustering")
    st.plotly_chart(make_3d_figure(points[spatial_cols].to_numpy(), axis_names,
                                   np.zeros(len(points), dtype=int), "Input points", aspectmode))

# --- Single algorithm tab ---------------------------------------------------
with tab_single:
    name = st.selectbox("Algorithm", ALGORITHMS)
    st.caption(f"Parameters (change in the sidebar): {format_params(params[name])}")
    signature = (data_signature, name, tuple(params[name].items()))
    if st.button("Run clustering", type="primary"):
        with st.spinner(f"Running {name}..."):
            labels, runtime = run_algorithm(name, X, params[name], seed)
            metrics = compute_metrics(X, labels, seed)
        st.session_state["single"] = {"signature": signature, "name": name, "labels": labels,
                                      "runtime": runtime, "metrics": metrics}

    result = st.session_state.get("single")
    if result is not None and result["signature"][0] == data_signature:
        if result["signature"] != signature:
            st.warning("Settings changed since this result was computed. Press **Run clustering** to update.")
        m = result["metrics"]
        c = st.columns(5)
        c[0].metric("Clusters", m["clusters"])
        c[1].metric("Noise", f"{m['noise_fraction']:.1%}")
        c[2].metric("Silhouette ↑", f"{m['silhouette']:.3f}")
        c[3].metric("Davies-Bouldin ↓", f"{m['davies_bouldin']:.3f}")
        c[4].metric("Runtime", f"{result['runtime']:.2f} s")
        st.plotly_chart(make_3d_figure(coords, plot_axes, result["labels"], result["name"], plot_aspect))
        # Cluster profile: where each cluster sits and its mean intensity (e.g. mean HU = tissue type)
        profile = points.assign(cluster=result["labels"]).groupby("cluster").agg(
            size=(spatial_cols[0], "size"), **{f"mean_{col}": (col, "mean") for col in points.columns})
        st.dataframe(profile.round(1))
    elif result is not None:
        st.info("The data or features changed. Press **Run clustering** to compute a new result.")

# --- Compare tab ----------------------------------------------------------
with tab_compare:
    chosen = st.multiselect("Algorithms to compare", ALGORITHMS, default=ALGORITHMS)
    st.caption("All algorithms run on exactly the same points and features, so the metrics are comparable.")
    compare_signature = (data_signature, tuple((a, tuple(params[a].items())) for a in chosen))
    if st.button("Run comparison", type="primary", disabled=not chosen):
        rows, labels_by_algo = [], {}
        progress = st.progress(0.0)
        for i, algo in enumerate(chosen):
            progress.progress(i / len(chosen), text=f"Running {algo}...")
            labels, runtime = run_algorithm(algo, X, params[algo], seed)
            metrics = compute_metrics(X, labels, seed)
            labels_by_algo[algo] = labels
            rows.append({"algorithm": algo, "parameters": format_params(params[algo]),
                         **metrics, "runtime_s": runtime})
        progress.empty()
        st.session_state["compare"] = {"signature": compare_signature,
                                       "table": pd.DataFrame(rows).set_index("algorithm"),
                                       "labels": labels_by_algo}

    result = st.session_state.get("compare")
    if result is not None and result["signature"][0] == data_signature:
        if result["signature"] != compare_signature:
            st.warning("Settings changed since this comparison. Press **Run comparison** to update.")
        table = result["table"]
        st.subheader("Quantitative comparison")
        st.caption("Silhouette: higher is better (−1 to 1). Davies-Bouldin: lower is better. "
                   "Calinski-Harabasz: higher is better. Noise points (DBSCAN) are excluded from the metrics.")
        st.dataframe(table, column_config={
            "noise_fraction": st.column_config.NumberColumn("noise", format="percent"),
            "silhouette": st.column_config.NumberColumn("silhouette ↑", format="%.3f"),
            "davies_bouldin": st.column_config.NumberColumn("Davies-Bouldin ↓", format="%.3f"),
            "calinski_harabasz": st.column_config.NumberColumn("Calinski-Harabasz ↑", format="%.0f"),
            "runtime_s": st.column_config.NumberColumn("runtime (s)", format="%.2f"),
        })
        # Name the winner per metric; NaN (fewer than 2 clusters) is skipped by idxmax/idxmin
        best = []
        for column, label, pick in [("silhouette", "Silhouette", "max"), ("davies_bouldin", "Davies-Bouldin", "min"),
                                    ("calinski_harabasz", "Calinski-Harabasz", "max"), ("runtime_s", "Fastest", "min")]:
            values = table[column].dropna()
            if not values.empty:
                best.append(f"**{label}:** {values.idxmax() if pick == 'max' else values.idxmin()}")
        st.markdown(" · ".join(best))

        st.subheader("Visual comparison")
        figures = {}
        grid = st.columns(2)
        for i, (algo, labels) in enumerate(result["labels"].items()):
            fig = make_3d_figure(coords, plot_axes, labels,
                                 f"{algo} ({table.loc[algo, 'clusters']} clusters)", plot_aspect, height=480)
            figures[algo] = fig
            grid[i % 2].plotly_chart(fig)

        # Exports for the presentation: the table as CSV, and all plots in one interactive HTML file
        html_parts = ["<html><head><meta charset='utf-8'><title>Clustering comparison</title></head><body>",
                      f"<h1>Clustering comparison: {preview[-1]}</h1>",
                      f"<p>{len(points):,} points, features: {', '.join(feature_cols)}, "
                      f"PCA: {f'{n_components} components' if use_pca else 'off'}</p>",
                      table.to_html(float_format=lambda v: f"{v:.3f}")]
        for i, fig in enumerate(figures.values()):
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
        html_parts.append("</body></html>")
        d1, d2 = st.columns(2)
        d1.download_button("Download metrics (CSV)", table.to_csv().encode(), "clustering_metrics.csv",
                           "text/csv")
        d2.download_button("Download report with 3D plots (HTML)", "\n".join(html_parts).encode(),
                           "clustering_report.html", "text/html")
    elif result is not None:
        st.info("The data or features changed. Press **Run comparison** to compute a new result.")
