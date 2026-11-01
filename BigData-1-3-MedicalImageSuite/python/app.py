"""Medical Image Analysis Suite – Python GUI (Streamlit).

Load a dataset, run the pipeline (normalize -> smooth -> PCA -> clustering /
segmentation), compare the algorithms, and compare the implementations:
own NumPy code, library code and the JavaScript implementation (via Node).

Run from this folder:   streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from mia import compare, dataio, library, own
from mia.js_bridge import node_available, run_js
from mia.phantom import GROUND_TRUTH_NAMES, make_phantom
from mia.pipeline import IMPLEMENTATIONS, METHODS, default_config, run_pipeline

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PHANTOM = "Synthetic phantom (with ground truth)"
# Same colours as PALETTE in js/app.js, so overlays look the same in both apps
PALETTE = np.array([
    [0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128],
], dtype=np.float64)

st.set_page_config(page_title="Medical Image Analysis Suite", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Generating phantom...")
def cached_phantom():
    return make_phantom()


@st.cache_resource(show_spinner="Reading DICOM files...", max_entries=2)
def cached_dicom_series(key, _data):
    return dataio.dicom_series(key, _data)


@st.cache_resource(show_spinner="Building volume...", max_entries=4)
def cached_dicom_volume(key, _datasets):
    return dataio.dicom_volume(_datasets)


def load_bytes(name, data):
    """Returns (volume, ground_truth or None) for an uploaded or local file."""
    lower = name.lower()
    if lower.endswith(".npy"):
        return dataio.load_npy(data), None
    if lower.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
        return dataio.load_image(data), None
    series = cached_dicom_series(name, data)
    labels = [s["label"] for s in series]
    choice = st.sidebar.selectbox("DICOM series", range(len(series)), format_func=lambda i: labels[i])
    return cached_dicom_volume(f"{name}:{choice}", series[choice]["datasets"]), None


def select_dataset():
    st.sidebar.header("1 · Dataset")
    local = sorted(p.name for p in DATA_DIR.glob("*") if p.suffix.lower() in
                   {".npy", ".zip", ".dcm", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}) if DATA_DIR.exists() else []
    source = st.sidebar.radio("Source", [PHANTOM, "Upload file"] + [f"data/{n}" for n in local])
    if source == PHANTOM:
        volume, gt = cached_phantom()
        return PHANTOM, volume, gt
    if source == "Upload file":
        file = st.sidebar.file_uploader("DICOM (.dcm / .zip), NumPy (.npy) or image",
                                        type=["dcm", "zip", "npy", "png", "jpg", "jpeg", "tif", "tiff", "bmp"])
        if file is None:
            st.info("Upload a file in the sidebar, or choose the synthetic phantom.")
            st.stop()
        return (file.name, *load_bytes(file.name, file.getvalue()))
    path = DATA_DIR / source.split("/", 1)[1]
    return (path.name, *load_bytes(path.name, path.read_bytes()))


# ---------------------------------------------------------------------------
# Sidebar: pipeline configuration (the same JSON as the JavaScript app)
# ---------------------------------------------------------------------------

def config_sidebar(volume):
    st.sidebar.header("2 · Pipeline")
    uploaded = st.sidebar.file_uploader("Import config (.json, e.g. from the JavaScript app)", type=["json"])
    base = default_config()
    if uploaded is not None:
        base.update(json.loads(uploaded.getvalue()))
    slices, H, W = volume.shape
    c = default_config()

    c["slice"] = st.sidebar.slider("Slice", 0, slices - 1, min(base["slice"], slices - 1)) if slices > 1 else 0
    with st.sidebar.expander("Pre-processing", expanded=True):
        lo, hi = st.slider("Normalization percentiles", 0.0, 100.0,
                           (float(base["normalize"]["p_low"]), float(base["normalize"]["p_high"])), 0.5)
        c["normalize"] = {"p_low": lo, "p_high": hi}
        c["smooth_sigma"] = st.slider("Gaussian smoothing σ (0 = off)", 0.0, 5.0, float(base["smooth_sigma"]), 0.25)
        c["pca"]["enabled"] = st.checkbox("PCA on 5 per-pixel features", base["pca"]["enabled"])
        c["pca"]["n_components"] = st.slider("PCA components", 1, 5, int(base["pca"]["n_components"]),
                                             disabled=not c["pca"]["enabled"])

    c["method"] = st.sidebar.selectbox("Method", list(METHODS), list(METHODS).index(base["method"]),
                                       format_func=METHODS.get)
    with st.sidebar.expander("Method parameters", expanded=True):
        st.caption("Values refer to the work image, scaled to [0, 1].")
        b = base["kmeans"]
        st.markdown("**K-Means**")
        c["kmeans"] = {"k": st.slider("k (clusters)", 2, 10, int(b["k"])),
                       "max_iter": st.number_input("Max iterations", 1, 1000, int(b["max_iter"])),
                       "tol": float(b["tol"])}
        b = base["threshold"]
        st.markdown("**Thresholding**")
        mode = st.radio("Threshold", ["otsu", "manual"], ["otsu", "manual"].index(b["mode"]), horizontal=True,
                        format_func={"otsu": "Otsu (automatic)", "manual": "Manual"}.get)
        c["threshold"] = {"mode": mode, "value": st.slider("Manual threshold", 0.0, 1.0, float(b["value"]), 0.01,
                                                           disabled=mode == "otsu")}
        b = base["region_growing"]
        st.markdown("**Region growing**")
        c["region_growing"] = {"seed_x": st.slider("Seed x", 0, W - 1, min(int(b["seed_x"]), W - 1)),
                               "seed_y": st.slider("Seed y", 0, H - 1, min(int(b["seed_y"]), H - 1)),
                               "tolerance": st.slider("Tolerance", 0.0, 0.5, float(b["tolerance"]), 0.01)}
        b = base["watershed"]
        st.markdown("**Watershed markers**")
        low, high = st.slider("Background < low, objects > high", 0.0, 1.0, (float(b["low"]), float(b["high"])), 0.01)
        c["watershed"] = {"low": low, "high": high,
                          "min_marker_size": st.number_input("Min. marker size (px)", 1, 10000,
                                                             int(b["min_marker_size"]))}
    return c


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def to_uint8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def overlay(grey, labels, alpha=0.45):
    """Grey image with coloured labels on top; label 0 stays uncoloured."""
    rgb = np.repeat(np.clip(grey, 0, 1)[..., None], 3, axis=2) * 255
    colours = PALETTE[np.where(labels > 0, (labels - 1) % (len(PALETTE) - 1) + 1, 0)]
    mask = (labels > 0)[..., None]
    return np.where(mask, (1 - alpha) * rgb + alpha * colours, rgb).astype(np.uint8)


def kmeans_overlay(grey, labels, alpha=0.45):
    """K-Means labels start at 0, so shift them by one to colour every cluster."""
    return overlay(grey, labels + 1, alpha)


def result_image(grey, labels, method):
    return kmeans_overlay(grey, labels) if method == "kmeans" else overlay(grey, labels)


def mark_seed(image, config):
    if config["method"] != "region_growing":
        return image
    image = image.copy()
    y, x = config["region_growing"]["seed_y"], config["region_growing"]["seed_x"]
    H, W = image.shape[:2]
    image[max(y - 3, 0):y + 4, min(max(x, 0), W - 1)] = [255, 255, 0]
    image[min(max(y, 0), H - 1), max(x - 3, 0):x + 4] = [255, 255, 0]
    return image


def display_slice(img, config):
    norm, _, _ = own.normalize(img.astype(np.float64), config["normalize"]["p_low"], config["normalize"]["p_high"])
    return norm


def timings_frame(rows):
    return pd.DataFrame(rows).set_index("implementation").round(2)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def tab_pipeline(name, volume, gt, config, impl_name):
    img = volume[config["slice"]]
    result = run_pipeline(img, config, IMPLEMENTATIONS[impl_name])
    grey = display_slice(img, config)

    c1, c2, c3 = st.columns(3)
    c1.image(mark_seed(np.repeat(to_uint8(grey)[..., None], 3, axis=2), config),
             caption=f"Input – slice {config['slice']} (normalized)", width="stretch")
    c2.image(mark_seed(np.repeat(to_uint8(result["work"])[..., None], 3, axis=2), config),
             caption="Work image (" + ("PC1" if config["pca"]["enabled"] else "smoothed") + ")",
             width="stretch")
    c3.image(result_image(grey, result["labels"], config["method"]),
             caption=f"{METHODS[config['method']]} – {impl_name}", width="stretch")

    stats = compare.label_stats(result["labels"])
    m = st.columns(5)
    m[0].metric("Total runtime", f"{result['timings']['total']:.1f} ms")
    m[1].metric("Labels", stats["labels"])
    m[2].metric("Foreground", f"{stats['foreground %']:.1f} %")
    if gt is not None:
        a = compare.agreement(gt[config["slice"]], result["labels"])
        m[3].metric("ARI vs ground truth", f"{a['ARI']:.3f}")
        m[4].metric("Matched accuracy vs GT", f"{a['matched agreement']:.1%}")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Runtime per step (ms)**")
        st.dataframe(pd.DataFrame([result["timings"]]).round(2), hide_index=True)
        st.markdown("**Method info**")
        st.json(result["info"], expanded=False)
    with right:
        if config["pca"]["enabled"]:
            st.markdown("**PCA**")
            explained = result["info"]["explained_variance"]
            # Plotly instead of st.bar_chart: st.bar_chart needs Altair, which fails
            # with the older typing_extensions in some Anaconda installations
            st.plotly_chart(px.bar(x=[f"PC{i + 1}" for i in range(len(explained))], y=explained,
                                   labels={"x": "component", "y": "explained variance ratio"}, height=260))
            st.dataframe(pd.DataFrame(result["components"], columns=own.FEATURE_NAMES,
                                      index=[f"PC{i + 1}" for i in range(len(explained))]).round(3))
    if config["pca"]["enabled"]:
        cols = st.columns(len(result["pca_images"]))
        for i, (col, pc) in enumerate(zip(cols, result["pca_images"])):
            col.image(to_uint8(pc), caption=f"PC{i + 1}", width="stretch")
    if gt is not None:
        with st.expander("Ground truth of the phantom"):
            st.image(overlay(grey, gt[config["slice"]].astype(np.int32)), width=300,
                     caption=" · ".join(f"{i} {n}" for i, n in enumerate(GROUND_TRUTH_NAMES)))

    st.markdown("**Export (shared format, can be opened in the JavaScript app)**")
    d = st.columns(4)
    d[0].download_button("labels.npy", dataio.npy_bytes(result["labels"]), "labels.npy")
    d[1].download_button("result.json", dataio.result_json(config, result, impl_name, name), "result.json")
    d[2].download_button("config.json", json.dumps(config, indent=2), "config.json")
    # The whole volume is only offered when small; serializing a 100+ MB MR
    # series on every rerun would make the app slow
    if volume.nbytes <= 64 * 1024 ** 2:
        d[3].download_button("volume.npy (dataset)", dataio.npy_bytes(volume), "volume.npy")
    else:
        d[3].download_button("slice.npy (dataset)", dataio.npy_bytes(img.astype(np.float32)), "slice.npy",
                             help="Volume too large to download; this is the current slice (use slice 0 in JS)")


def method_variants(config, with_without_pca):
    """All methods with the current parameters; threshold both as Otsu and manual."""
    variants = []
    pca_options = [False, True] if with_without_pca else [config["pca"]["enabled"]]
    for use_pca in pca_options:
        for method in METHODS:
            modes = ["otsu", "manual"] if method == "threshold" else [None]
            for mode in modes:
                c = json.loads(json.dumps(config))
                c["method"] = method
                c["pca"]["enabled"] = use_pca
                label = METHODS[method]
                if mode:
                    c["threshold"]["mode"] = mode
                    label += f" ({mode})"
                variants.append((label + (" + PCA" if use_pca else ""), c))
    return variants


def tab_algorithms(volume, gt, config, impl_name):
    st.markdown("Runs every method on the same slice with the parameters from the sidebar.")
    both = st.checkbox("Run each method with and without PCA", True)
    if not st.button("Run all algorithms", type="primary"):
        return
    img = volume[config["slice"]]
    grey = display_slice(img, config)
    rows, images = [], []
    progress = st.progress(0.0)
    variants = method_variants(config, both)
    for i, (label, c) in enumerate(variants):
        r = run_pipeline(img, c, IMPLEMENTATIONS[impl_name])
        row = {"method": label, **compare.label_stats(r["labels"]),
               "runtime ms": r["timings"]["total"], "method ms": r["timings"]["method"]}
        if gt is not None:
            a = compare.agreement(gt[config["slice"]], r["labels"])
            row["ARI vs GT"] = a["ARI"]
            row["accuracy vs GT"] = a["matched agreement"]
        rows.append(row)
        images.append((label, result_image(grey, r["labels"], c["method"])))
        progress.progress((i + 1) / len(variants))
    table = pd.DataFrame(rows)
    st.dataframe(table.round(4), hide_index=True, width="stretch")
    if gt is not None:
        best = table.sort_values("ARI vs GT", ascending=False).iloc[0]
        st.success(f"Best agreement with ground truth: **{best['method']}** (ARI {best['ARI vs GT']:.3f})")
    st.plotly_chart(px.bar(table, x="method", y="runtime ms", title=f"Runtime per method – {impl_name}"))
    cols = st.columns(4)
    for i, (label, image) in enumerate(images):
        cols[i % 4].image(image, caption=label, width="stretch")
    st.download_button("Download table (CSV)", table.to_csv(index=False), "algorithm_comparison.csv")


def tab_implementations(volume, gt, config):
    st.markdown("Runs the **current method** with each implementation on the same slice and compares "
                "label maps (matched pixel agreement, ARI) and runtime. JavaScript runs in Node.js "
                "with exactly the same input and config.")
    repeats = st.slider("Repeats per implementation (runtime = median)", 1, 10, 3)
    use_js = st.checkbox("Include JavaScript (Node.js)", node_available(), disabled=not node_available(),
                         help=None if node_available() else "Node.js was not found on PATH")
    browser_file = st.file_uploader("Optional: labels.npy exported from the JavaScript web app", type=["npy"])
    if not st.button("Compare implementations", type="primary"):
        return

    img = volume[config["slice"]]
    grey = display_slice(img, config)
    results, time_rows = {}, []
    for name, impl in IMPLEMENTATIONS.items():
        runs = [run_pipeline(img, config, impl) for _ in range(repeats)]
        results[name] = (runs[-1]["labels"], runs[-1]["work"])
        time_rows.append({"implementation": name,
                          **{k: float(np.median([r["timings"][k] for r in runs])) for k in runs[0]["timings"]}})
    if use_js:
        with st.spinner("Running JavaScript implementation in Node.js..."):
            labels, work, js = run_js(img, config, repeats)
        name = js["implementation"]
        results[name] = (labels, work)
        time_rows.append({"implementation": name,
                          **{k: float(np.median([r[k] for r in js["runs_ms"]])) for k in js["runs_ms"][0]}})
    if browser_file is not None:
        labels = np.load(browser_file, allow_pickle=False).astype(np.int32)
        if labels.shape != img.shape:
            st.error(f"Uploaded labels have shape {labels.shape}, slice has {img.shape}.")
        else:
            results["JavaScript (browser export)"] = (labels, None)

    st.markdown("**Runtime per step (median, ms)**")
    times = timings_frame(time_rows)
    st.dataframe(times, width="stretch")
    long = times.drop(columns="total").reset_index().melt("implementation", var_name="step", value_name="ms")
    st.plotly_chart(px.bar(long, x="implementation", y="ms", color="step", title="Runtime per pipeline step"))

    reference = "Python (own NumPy)"
    ref_labels, ref_work = results[reference]
    rows = []
    for name, (labels, work) in results.items():
        a = compare.agreement(ref_labels, labels)
        row = {"implementation": name, "matched agreement vs own": a["matched agreement"],
               "ARI vs own": a["ARI"], "differing pixels": a["differing pixels"],
               "max |Δ work image|": float(np.abs(work - ref_work).max()) if work is not None else None}
        if gt is not None:
            row["ARI vs GT"] = compare.agreement(gt[config["slice"]], labels)["ARI"]
        rows.append(row)
    st.markdown(f"**Agreement with {reference}**")
    st.dataframe(pd.DataFrame(rows).set_index("implementation"), width="stretch")

    cols = st.columns(len(results))
    for col, (name, (labels, _)) in zip(cols, results.items()):
        col.image(result_image(grey, labels, config["method"]), caption=name, width="stretch")
        diff = compare.agreement(ref_labels, labels)["difference_mask"]
        col.image(overlay(grey, diff.astype(np.int32), alpha=0.9),
                  caption=f"Differences vs own ({int(diff.sum())} px, red)", width="stretch")


def main():
    st.title("Medical Image Analysis Suite")
    st.caption("Pipeline: normalize → Gaussian smoothing → PCA → K-Means / thresholding / region growing / "
               "watershed. Same algorithms implemented in Python and JavaScript (see SPEC.md).")
    name, volume, gt = select_dataset()
    config = config_sidebar(volume)
    st.sidebar.header("3 · Implementation")
    impl_name = st.sidebar.radio("Used in the first two tabs", list(IMPLEMENTATIONS))
    st.sidebar.caption(f"Dataset: {name} · shape {volume.shape}")

    t1, t2, t3 = st.tabs(["Pipeline", "Compare algorithms", "Compare implementations"])
    with t1:
        tab_pipeline(name, volume, gt, config, impl_name)
    with t2:
        tab_algorithms(volume, gt, config, impl_name)
    with t3:
        tab_implementations(volume, gt, config)


main()
