"""Streamlit interface for comparing the notebook's lossless coders."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from benchmark import check_requirements, peak_memory, time_function
from coders import CODERS, entropy


ROOT = Path(__file__).parent
DATA_CANDIDATES = (
    ROOT / "insurance.csv",
    ROOT.parent / "Tarjei/smarttechbigdata/SmartTech-1-2b-HuffmanGUI/insurance.csv",
    ROOT.parent / "smarttechbigdata/SmartTech-1-2b-HuffmanGUI/insurance.csv",
)
SCALE_FACTORS = {"x1": 1, "x2": 2, "x4": 4, "x8": 8}


def find_default_dataset():
    """Find the default example beside the app or in the original repository location."""
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in DATA_CANDIDATES)
    raise FileNotFoundError(f"Could not find insurance.csv. Searched:\n{searched}")


def load_dataset(uploaded_file):
    """Read Excel input as CSV bytes, or return the exact default example bytes."""
    if uploaded_file is None:
        default_path = find_default_dataset()
        data = default_path.read_bytes()
        frame = pd.read_csv(default_path)
        return data, frame
    frame = pd.read_excel(uploaded_file)
    data = frame.to_csv(index=False).encode("utf-8")
    return data, frame


def status_mark(passed):
    """Use visual status marks so requirement results scan quickly."""
    return "✓" if passed else "✗"


def run_one(name, encode, decode, data, max_seconds, max_memory_factor, progress_bar):
    """Run one coder in order, measuring time and memory in separate phases."""
    def encode_progress(fraction):
        progress_bar.progress(min(0.5 * fraction, 1.0), text=f"{name}: encoding")

    def decode_progress(fraction):
        progress_bar.progress(0.5 + 0.5 * fraction, text=f"{name}: decoding")

    encode_time, compressed = time_function(lambda value: encode(value, progress=encode_progress), data)
    decode_time, decoded = time_function(lambda value: decode(value, progress=decode_progress), compressed)
    encode_peak, _ = peak_memory(encode, data)
    decode_peak, _ = peak_memory(decode, compressed)
    peak_factor = max(encode_peak, decode_peak) / len(data)
    checks = check_requirements(
        encode_time,
        decode_time,
        peak_factor,
        decoded,
        data,
        max_seconds,
        max_memory_factor,
    )
    progress_bar.progress(1.0, text=f"{name}: complete")
    return {
        "Algorithm": name,
        "Compressed size (bytes)": len(compressed),
        "Ratio (%)": 100 * len(compressed) / len(data),
        "Bits per byte": 8 * len(compressed) / len(data),
        "Encode time (s)": encode_time,
        "Decode time (s)": decode_time,
        "Peak memory (x input)": peak_factor,
        "Speed K1": status_mark(checks["Speed K1"]),
        "Memory K2": status_mark(checks["Memory K2"]),
        "Lossless K3": status_mark(checks["Lossless K3"]),
    }


def plot_results(results, minimum_bytes):
    """Draw the requested size and runtime comparisons from one result table."""
    size_figure, size_axis = plt.subplots()
    size_axis.bar(results["Algorithm"], results["Compressed size (bytes)"], color="#176b87")
    size_axis.axhline(minimum_bytes, color="#d1495b", linestyle="--", label="Entropy minimum")
    size_axis.set_ylabel("Compressed size (bytes)")
    size_axis.tick_params(axis="x", rotation=25)
    size_axis.legend()
    st.pyplot(size_figure, clear_figure=True)

    time_figure, time_axis = plt.subplots()
    positions = range(len(results))
    width = 0.38
    time_axis.bar([position - width / 2 for position in positions], results["Encode time (s)"], width, label="Encode")
    time_axis.bar([position + width / 2 for position in positions], results["Decode time (s)"], width, label="Decode")
    time_axis.set_xticks(list(positions), results["Algorithm"], rotation=25)
    time_axis.set_ylabel("Time (s)")
    time_axis.legend()
    st.pyplot(time_figure, clear_figure=True)


def plot_scaling(scale_results):
    """Plot encode time against repeated input size like the notebook."""
    figure, axis = plt.subplots()
    for name, points in scale_results.items():
        axis.plot(points["size_mb"], points["encode_time"], marker="o", label=name)
    axis.set_xlabel("Input size (MB)")
    axis.set_ylabel("Encode time (s)")
    axis.set_title("Encode time vs input size")
    axis.legend()
    st.pyplot(figure, clear_figure=True)


st.set_page_config(page_title="Lossless Compression Lab", layout="wide")
st.title("Lossless Compression Lab")
st.caption("The algorithms and benchmark definitions are reused from Compression.ipynb.")

with st.sidebar:
    st.header("Benchmark settings")
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    selected_names = [name for name in CODERS if st.checkbox(name, value=True)]
    max_seconds = st.number_input(
        "K1 max seconds",
        min_value=0.0,
        value=10.0,
        step=1.0,
        help="K1 passes when encode time plus decode time is at or below this limit.",
    )
    max_memory_factor = st.number_input(
        "K2 max memory factor",
        min_value=0.0,
        value=50.0,
        step=5.0,
        help="K2 passes when peak memory is no more than this multiple of the input size.",
    )
    st.checkbox(
        "K3: Lossless roundtrip",
        value=True,
        disabled=True,
        help="K3 passes only when decoding the compressed bytes reproduces the input exactly.",
    )
    scale_label = st.selectbox("Scale factor", list(SCALE_FACTORS))
    run_benchmark = st.button("Run benchmark", type="primary")

try:
    base_data, base_frame = load_dataset(uploaded_file)
except Exception as error:
    st.error(f"Could not read the dataset: {error}")
    st.stop()

data = base_data * SCALE_FACTORS[scale_label]

st.header("Dataset")
preview_column, statistics_column = st.columns([1, 1])
with preview_column:
    st.dataframe(base_frame.head(), use_container_width=True)
with statistics_column:
    data_entropy = entropy(data)
    st.metric("Size", f"{len(data):,} bytes")
    st.metric("Distinct byte values", len(set(data)))
    st.metric("Entropy", f"{data_entropy:.3f} bits/byte")
    st.metric("Theoretical minimum", f"{data_entropy * len(data) / 8:,.0f} bytes")

if run_benchmark:
    if not selected_names:
        st.error("Select at least one algorithm.")
        st.stop()

    progress_bars = {name: st.progress(0, text=f"{name}: waiting") for name in selected_names}
    result_rows = []
    for name in selected_names:
        encode, decode = CODERS[name]
        with st.spinner(f"{name} is running. Please wait..."):
            result_rows.append(
                run_one(name, encode, decode, data, max_seconds, max_memory_factor, progress_bars[name])
            )
    results = pd.DataFrame(result_rows)
    st.session_state["results"] = results
    st.session_state["minimum_bytes"] = data_entropy * len(data) / 8

    # Scaling timings use one run per point so the requested chart stays practical.
    scale_results = {name: {"size_mb": [], "encode_time": []} for name in selected_names}
    for factor in SCALE_FACTORS.values():
        scaled_data = base_data * factor
        for name in selected_names:
            encode, _ = CODERS[name]
            encode_time, _ = time_function(encode, scaled_data, repeats=1)
            scale_results[name]["size_mb"].append(len(scaled_data) / 1_000_000)
            scale_results[name]["encode_time"].append(encode_time)
    st.session_state["scale_results"] = scale_results

if "results" in st.session_state:
    st.header("Results")
    results = st.session_state["results"]
    try:
        results_display = results.style.map(
            lambda value: "color: #16803c" if value == "✓" else "color: #c62828"
            if value == "✗" else ""
        )
    except AttributeError:
        # Keep the numeric results visible if an alternate environment lacks Styler support.
        results_display = results
    st.dataframe(
        results_display,
        use_container_width=True,
        hide_index=True,
    )
    st.header("Charts")
    plot_results(st.session_state["results"], st.session_state["minimum_bytes"])
    plot_scaling(st.session_state["scale_results"])
else:
    st.info("Choose the settings and click Run benchmark to start.")
