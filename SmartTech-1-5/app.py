"""Data Coder: compare Huffman, arithmetic coding and CABAC on image data.

Run with: streamlit run app.py
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import benchmark
import prediction
import sources
from coders import CODERS, arithmetic, cabac, huffman

# Colour follows the coder across every chart. Validated categorical slots 1-3.
COLORS = {"Huffman": "#2a78d6", "Arithmetic": "#eb6834", "CABAC": "#1baf7a"}
CONTEXT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#8a8983"
INK = "#52514e"
GRID = "#e8e7e2"
SURFACE = "#fcfcfb"
TABLE_ROWS = 200

st.set_page_config(page_title="Data Coder: Huffman, Arithmetic, CABAC", layout="wide")


# --- Cached loaders and computations ----------------------------------------------

@st.cache_resource(show_spinner=False)
def read_archive(path: str) -> bytes:
    """A resource, not data: Streamlit must not copy 60 MB on every rerun."""
    with open(path, "rb") as handle:
        return handle.read()


# `archive_id` identifies the archive; the underscore stops Streamlit hashing the bytes.
@st.cache_data(show_spinner=False)
def dicom_members(_archive: bytes, archive_id: str) -> list[str]:
    return sources.list_dicom_members(_archive)


@st.cache_data(show_spinner=False)
def dicom_slice(_archive: bytes, archive_id: str, member: str, representation: str) -> sources.Source:
    data = sources.read_dicom_member(_archive, member)
    return sources.dicom_source(data, f"Harnverhalt {member.rsplit('/', 1)[-1]}", representation)


@st.cache_data(show_spinner=False)
def load_generated(name: str, size: int, seed: int) -> sources.Source:
    return sources.generated(name, size, seed)


@st.cache_data(show_spinner=False)
def load_sample(name: str) -> sources.Source:
    return sources.sample_photo(name)


@st.cache_data(show_spinner=False)
def run_coders(pixels: np.ndarray, predictor: str, nominal_bits: int) -> list[benchmark.CoderResult]:
    return benchmark.run_all(pixels, predictor, nominal_bits)


@st.cache_data(show_spinner=False)
def huffman_table(symbols: np.ndarray) -> dict[int, str]:
    return huffman.code_table(symbols)


@st.cache_data(show_spinner=False)
def arithmetic_trace(symbols: np.ndarray, limit: int) -> list[dict]:
    return arithmetic.trace(symbols, limit)


@st.cache_data(show_spinner=False)
def cabac_trace(symbols: np.ndarray, limit: int, mode: int) -> list[dict]:
    return cabac.trace(symbols, limit, mode)


# --- Display helpers --------------------------------------------------------------

def to_display(values: np.ndarray) -> np.ndarray:
    """Stretch to 8 bits for the screen. The coders never see this."""
    low, high = float(values.min()), float(values.max())
    if high == low:
        return np.full(values.shape, 128, dtype=np.uint8)
    return np.round((values - low) / (high - low) * 255).astype(np.uint8)


def residual_display(residuals: np.ndarray) -> np.ndarray:
    """Zero residual is mid-gray; the 99th percentile sets the contrast."""
    scale = max(1.0, float(np.percentile(np.abs(residuals), 99)))
    return np.round(np.clip(residuals / scale, -1, 1) * 127 + 128).astype(np.uint8)


def format_bytes(count: float) -> str:
    return f"{count:,.0f} B" if count < 10_000 else f"{count / 1024:,.1f} KiB"


def format_seconds(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def chart_layout(figure: go.Figure, title: str, x: str, y: str, height: int = 320) -> go.Figure:
    figure.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0b0b0b"), x=0, xanchor="left"),
        height=height,
        margin=dict(l=56, r=24, t=48, b=48),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(color=INK, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_color="#0b0b0b"),
    )
    figure.update_xaxes(title=x, gridcolor=GRID, zeroline=False, linecolor=GRID)
    figure.update_yaxes(title=y, gridcolor=GRID, zeroline=False, linecolor=GRID)
    return figure


def show_chart(figure: go.Figure) -> None:
    st.plotly_chart(figure, use_container_width=True, theme=None, config={"displaylogo": False})


def show_table(frame: pd.DataFrame, **options) -> None:
    st.dataframe(frame, width="stretch", hide_index=True, **options)


def coder_bars(results: list[benchmark.CoderResult], value, title: str, axis: str, text) -> go.Figure:
    names = [result.coder for result in results]
    values = [value(result) for result in results]
    figure = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=[COLORS[name] for name in names],
            text=[text(item) for item in values],
            textposition="outside",
            cliponaxis=False,
            width=0.5,
            hovertemplate="%{x}: %{text}<extra></extra>",
        )
    )
    figure.update_yaxes(range=[0, max(values + [1e-9]) * 1.2])  # Room for the outside labels.
    return chart_layout(figure, title, "", axis, height=300)


def histogram_figure(values: np.ndarray, title: str, x_title: str) -> go.Figure:
    uniques, counts = np.unique(values, return_counts=True)
    if uniques.size > 256:
        counts, edges = np.histogram(values, bins=256)
        uniques = (edges[:-1] + edges[1:]) / 2
    figure = go.Figure(
        go.Bar(x=uniques, y=counts, marker_color=NEUTRAL, marker_line_width=0, hovertemplate="%{x}: %{y:,}<extra></extra>")
    )
    figure.update_layout(bargap=0)
    return chart_layout(figure, title, x_title, "Count", height=280)


# --- Sidebar: choose the data -----------------------------------------------------

def medical_source() -> sources.Source | None:
    upload = st.sidebar.file_uploader("Harnverhalt .zip or single .dcm (optional)", type=["zip", "dcm"])
    representation = st.sidebar.radio(
        "Representation",
        ["windowed", "stored"],
        format_func=lambda value: {"windowed": "Soft-tissue window (8 bit)", "stored": "Stored scanner values (12 bit)"}[value],
    )
    if upload is not None and upload.name.lower().endswith(".dcm"):
        return sources.dicom_source(upload.getvalue(), upload.name, representation)

    if upload is not None:
        archive = upload.getvalue()
        archive_id = f"upload:{upload.file_id}"
    else:
        path = sources.find_harnverhalt_zip()
        if path is None:
            st.sidebar.warning(f"{sources.HARNVERHALT_NAME} was not found. Upload it above, or put it in SmartTech-1-5/data/.")
            return None
        archive = read_archive(str(path))
        archive_id = str(path)

    members = dicom_members(archive, archive_id)
    if not members:
        st.sidebar.error("The archive contains no .dcm files.")
        return None
    number = st.sidebar.slider("Slice", 1, len(members), min(141, len(members)))
    return dicom_slice(archive, archive_id, members[number - 1], representation)


def choose_source() -> sources.Source | None:
    st.sidebar.header("1. Data")
    category = st.sidebar.radio("Source type", ["Medical (Harnverhalt)", "Real-life photo", "Generated", "Raw file"])

    if category == "Generated":
        name = st.sidebar.selectbox("Pattern", list(sources.GENERATORS))
        side = st.sidebar.select_slider("Side (px)", [64, 128, 256, 512], value=256)
        seed = st.sidebar.number_input("Seed", 0, 9999, 1)
        return load_generated(name, side, int(seed))

    if category == "Medical (Harnverhalt)":
        return medical_source()

    if category == "Real-life photo":
        upload = st.sidebar.file_uploader("Your own photo (optional)", type=["png", "jpg", "jpeg"])
        if upload is not None:
            return sources.photo_source(upload.getvalue(), upload.name)
        return load_sample(st.sidebar.selectbox("Sample photo", sources.list_samples()))

    upload = st.sidebar.file_uploader("Any file, read as unsigned samples")
    sample_bits = st.sidebar.radio("Sample size", [8, 16], horizontal=True, format_func=lambda bits: f"{bits} bit")
    big_endian = sample_bits == 16 and st.sidebar.checkbox("Big endian")
    width = st.sidebar.number_input("Width in samples (0 = one long row)", 0, 100_000, 0)
    if upload is None:
        st.info("Upload a file in the sidebar. Width > 0 enables the top-neighbour context and 2D prediction.")
        return None
    try:
        return sources.raw_source(upload.getvalue(), upload.name, sample_bits, int(width), big_endian)
    except ValueError as error:
        st.error(str(error))
        return None


st.title("Data Coder")
st.caption("Lossless entropy coding of image data: Huffman, arithmetic coding and CABAC, on the same symbols.")

source = choose_source()

st.sidebar.header("2. Processing")
max_side = st.sidebar.select_slider(
    "Crop to max side (px)",
    [64, 128, 256, 512, 1024],
    value=256,
    help="Center crop, because resampling would change the noise. The coders are pure Python: larger takes longer.",
)
predictor_label = st.sidebar.radio(
    "Prediction before coding",
    list(prediction.PREDICTORS),
    index=2,
    help="All coders receive the same symbols, so differences come from the coders alone.",
)
predictor = prediction.PREDICTORS[predictor_label]
trace_limit = st.sidebar.select_slider("Symbols to trace", [200, 500, 1000, 2000, 5000], value=2000)

if source is None:
    st.stop()

pixels = sources.limit_size(source.pixels, max_side)
symbols = prediction.forward(pixels, predictor)
entropy = benchmark.entropy_bits(symbols)
mode = benchmark.cabac_mode(predictor)

with st.spinner("Encoding and decoding with all three coders..."):
    results = run_coders(pixels, predictor, source.nominal_bits)
    huffman_codes = huffman_table(symbols)
    arithmetic_rows = arithmetic_trace(symbols, trace_limit)
    cabac_rows = cabac_trace(symbols, trace_limit, mode)

by_name = {result.coder: result for result in results}

tab_input, tab_results, tab_huffman, tab_arithmetic, tab_cabac, tab_matrix = st.tabs(
    ["Input", "Results", "Huffman", "Arithmetic", "CABAC", "Benchmark all data"]
)


# --- Input ------------------------------------------------------------------------

with tab_input:
    metrics = st.columns(4)
    metrics[0].metric("Size", f"{pixels.shape[1]} x {pixels.shape[0]}")
    metrics[1].metric("Nominal depth", f"{source.nominal_bits} bit")
    metrics[2].metric("Distinct symbols", f"{np.unique(symbols).size:,}")
    metrics[3].metric("Entropy H0 (bit/symbol)", f"{entropy:.3f}")
    st.caption(f"**{source.name}** ({source.category}). {source.note}")

    is_stream = pixels.shape[0] == 1
    if is_stream:
        st.info("This is a single-row stream, so there is no image to preview. Set a width to view it as an image.")
    else:
        left, right = st.columns(2)
        left.image(to_display(pixels), caption="Input values (stretched for display)", width=420)
        if predictor == "none":
            right.info("Prediction is off, so the coders receive the pixel values directly.")
        else:
            right.image(residual_display(symbols), caption=f"Residuals, {predictor_label} (gray = 0)", width=420)

    left, right = st.columns(2)
    with left:
        show_chart(histogram_figure(pixels.ravel(), "Input value histogram", "Value"))
    with right:
        show_chart(histogram_figure(symbols.ravel(), "Histogram of the coded symbols", "Symbol"))
    st.caption("A narrower right-hand histogram means lower H0: this is what prediction buys the entropy coders.")


# --- Results ----------------------------------------------------------------------

def result_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Coder": result.coder,
                "Total": format_bytes(result.total_bytes),
                "Header": format_bytes(result.header_bytes),
                "Bits/pixel": round(result.bits_per_pixel, 3),
                "Ratio": round(result.ratio, 3),
                "Saving": f"{result.saving_percent:.1f} %",
                "Above H0": "n/a" if math.isnan(result.gap_to_entropy_percent) else f"{result.gap_to_entropy_percent:+.1f} %",
                "Encode": format_seconds(result.encode_seconds),
                "Decode": format_seconds(result.decode_seconds),
                "Lossless": "Yes" if result.lossless else "NO",
            }
            for result in results
        ]
    )


def explain_result() -> str:
    """Turn the measurements into the sentence a viewer should take away."""
    best = min(results, key=lambda result: result.total_bytes)
    h, a, c = by_name["Huffman"], by_name["Arithmetic"], by_name["CABAC"]
    parts = [f"**{best.coder}** is smallest at {best.bits_per_pixel:.3f} bits/pixel."]
    if entropy < 1 or h.total_bytes > 1.1 * a.total_bytes:
        parts.append(
            f"Huffman needs at least 1 bit per symbol, so it spends {h.bits_per_pixel:.3f} where "
            f"arithmetic coding gets close to H0 ({a.bits_per_pixel:.3f} vs {entropy:.3f})."
        )
    else:
        parts.append(
            f"Huffman and arithmetic coding are close ({h.bits_per_pixel:.3f} vs {a.bits_per_pixel:.3f}) "
            "because no symbol here is frequent enough to need less than 1 bit."
        )
    if c.bits_per_pixel < 0.97 * a.bits_per_pixel:
        parts.append(
            f"CABAC goes {(1 - c.bits_per_pixel / a.bits_per_pixel) * 100:.0f} % below arithmetic coding: "
            "contexts exploit structure between neighbours that an order-0 model cannot see."
        )
    elif c.bits_per_pixel > 1.01 * a.bits_per_pixel:
        parts.append(
            "CABAC is larger here: neighbours say little about the next symbol, "
            "so it pays for learning and for its binarization without gaining from contexts."
        )
    else:
        parts.append("CABAC matches arithmetic coding here: contexts add little beyond the order-0 model.")
    return " ".join(parts)


def cumulative_figure() -> go.Figure:
    """Bits spent over the first symbols, excluding headers."""
    flat = symbols.ravel()[: trace_limit].tolist()
    steps = np.arange(1, len(flat) + 1)
    huffman_bits = np.cumsum([len(huffman_codes[symbol]) for symbol in flat])
    arithmetic_bits = np.array([row["ideal bits"] for row in arithmetic_rows])
    cabac_frame = pd.DataFrame(cabac_rows)
    cabac_bits = cabac_frame.groupby("symbol #")["total bits"].last().to_numpy()

    figure = go.Figure()
    for name, values in (("Huffman", huffman_bits), ("Arithmetic", arithmetic_bits), ("CABAC", cabac_bits)):
        figure.add_scatter(
            x=steps,
            y=values,
            name=name,
            mode="lines",
            line=dict(color=COLORS[name], width=2),
            hovertemplate=name + ": %{y:,.0f} bits after %{x:,} symbols<extra></extra>",
        )
    figure.update_layout(hovermode="x unified")
    return chart_layout(figure, f"Bits spent on the first {len(flat):,} symbols, without headers", "Symbols coded", "Bits", height=340)


with tab_results:
    if not all(result.lossless for result in results):
        st.error("Round trip FAILED for at least one coder.")
    else:
        st.success("Round trip verified: all coders decoded from bytes alone and restored every pixel.")

    st.markdown(explain_result())
    show_table(result_frame())

    charts = st.columns(3)
    with charts[0]:
        figure = coder_bars(results, lambda r: r.bits_per_pixel, f"Size (dotted: H0 = {entropy:.3f})", "Bits per pixel", lambda v: f"{v:.3f}")
        figure.add_hline(y=entropy, line=dict(color=INK, width=2, dash="dot"))
        show_chart(figure)
    with charts[1]:
        show_chart(coder_bars(results, lambda r: r.encode_seconds, "Encode time", "Seconds", format_seconds))
    with charts[2]:
        show_chart(coder_bars(results, lambda r: r.decode_seconds, "Decode time", "Seconds", format_seconds))
    st.caption(
        "Times are for pure-Python implementations on this machine. They show relative cost, not what a C codec would "
        "reach. Huffman's encode time is flattered: it joins precomputed strings, while the other two run per-symbol maths."
    )

    show_chart(cumulative_figure())
    st.caption(
        "Huffman and arithmetic coding use a model built from the whole image, sent in the header. "
        "CABAC starts knowing nothing, so it often lags early and catches up as its contexts learn. "
        "The arithmetic line is the ideal cost of its model; the CABAC line sums the real cost of each bin."
    )


# --- Huffman ----------------------------------------------------------------------

def payload_preview(result: benchmark.CoderResult) -> None:
    st.markdown("**Start of the encoded payload**")
    st.code(result.stream_head.hex(" ") or "(empty)", language=None, wrap_lines=True)


with tab_huffman:
    result = by_name["Huffman"]
    metrics = st.columns(4)
    metrics[0].metric("Alphabet", f"{result.stats.get('alphabet', 0):,}")
    metrics[1].metric("Mean code length", f"{result.stats.get('mean_code_length', 0):.3f} bit")
    metrics[2].metric("Longest code", f"{result.stats.get('max_code_length', 0)} bit")
    metrics[3].metric("Header (code lengths)", format_bytes(result.header_bytes))

    values, counts = np.unique(symbols, return_counts=True)
    table = pd.DataFrame({"Symbol": values, "Count": counts})
    table["p"] = table["Count"] / symbols.size
    table["-log2(p)"] = -np.log2(table["p"])
    table["Code"] = [huffman_codes[int(value)] for value in values]
    table["Length"] = table["Code"].str.len()
    table["Waste per use (bits)"] = table["Length"] - table["-log2(p)"]
    table = table.sort_values("Count", ascending=False)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Code table** (most frequent first)")
        show_table(table.head(TABLE_ROWS).round({"p": 5, "-log2(p)": 3, "Waste per use (bits)": 3}), height=340)
    with right:
        lengths = table.groupby("Length")["Count"].sum() / symbols.size * 100
        figure = go.Figure(
            go.Bar(
                x=lengths.index,
                y=lengths.values,
                marker_color=COLORS["Huffman"],
                hovertemplate="%{x} bit codes: %{y:.1f} % of symbols<extra></extra>",
            )
        )
        show_chart(chart_layout(figure, "Symbols by code length", "Code length (bits)", "% of symbols", height=340))

    st.markdown("**Worked example**: the first 12 symbols")
    first = symbols.ravel()[:12].tolist()
    example = {"": ["Symbol", "Code"]}
    for position, symbol in enumerate(first, start=1):
        example[str(position)] = [str(symbol), huffman_codes[symbol]]
    show_table(pd.DataFrame(example))
    st.caption(
        "\"Waste\" is the gap between a whole-bit code and the ideal -log2(p). "
        "It is the only thing arithmetic coding can win back from Huffman."
    )
    payload_preview(result)


# --- Arithmetic -------------------------------------------------------------------

with tab_arithmetic:
    result = by_name["Arithmetic"]
    metrics = st.columns(3)
    metrics[0].metric("Payload", format_bytes(result.payload_bytes))
    metrics[1].metric("Header (counts)", format_bytes(result.header_bytes))
    metrics[2].metric("Interval precision", f"{arithmetic.PRECISION} bit")

    frame = pd.DataFrame(arithmetic_rows)
    st.markdown("**Interval narrowing**, as it happens inside the integer coder")
    show_table(
        frame.head(TABLE_ROWS).round({"p": 5, "ideal bits": 2}),
        height=340,
    )
    st.caption(
        "low and high are shown before renormalization. When their leading bits agree, those bits are final and sent. "
        "Pending bits are the carry case: the interval straddles the middle, so the next bit is not known yet."
    )

    head = frame.head(min(200, len(frame)))
    figure = go.Figure()
    figure.add_scatter(
        x=head["step"],
        y=head["ideal bits"],
        name="Information so far, sum of -log2(p)",
        mode="lines",
        line=dict(color=NEUTRAL, width=2),
        hovertemplate="%{y:.1f} bits of information<extra></extra>",
    )
    figure.add_scatter(
        x=head["step"],
        y=head["bits sent"],
        name="Bits actually sent",
        mode="lines",
        line=dict(color=COLORS["Arithmetic"], width=2, shape="hv"),
        hovertemplate="%{y} bits sent<extra></extra>",
    )
    figure.update_layout(hovermode="x unified")
    show_chart(chart_layout(figure, "Output follows the information, regardless of symbol boundaries", "Symbols coded", "Bits"))
    payload_preview(result)


# --- CABAC ------------------------------------------------------------------------

def learning_figure(frame: pd.DataFrame) -> go.Figure:
    coded = frame[frame["context id"] >= 0]
    top = coded["context id"].value_counts().head(4).index
    figure = go.Figure()
    for color, context_id in zip(CONTEXT_COLORS, top):
        rows = coded[coded["context id"] == context_id]
        label = rows["context"].iloc[0]
        figure.add_scatter(
            x=np.arange(1, len(rows) + 1),
            y=[cabac.lps_probability(int(state)) for state in rows["state"]],
            name=label,
            mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=label + "<br>use %{x}: p(LPS) = %{y:.3f}<extra></extra>",
        )
    floor = cabac.lps_probability(cabac.LOWEST_ADAPTIVE_STATE)
    figure.add_hline(y=floor, line=dict(color=INK, width=1, dash="dot"), annotation_text=f"floor {floor:.4f}", annotation_position="bottom right")
    return chart_layout(figure, "How the busiest contexts learn", "Times this context was used", "Estimated p(LPS)", height=360)


with tab_cabac:
    result = by_name["CABAC"]
    stats = result.stats
    total_bins = stats.get("context_bins", 0) + stats.get("bypass_bins", 0)
    metrics = st.columns(4)
    metrics[0].metric("Bins per symbol", f"{stats.get('bins_per_symbol', 0):.2f}")
    metrics[1].metric("Bypass bins", f"{stats.get('bypass_bins', 0) / max(total_bins, 1) * 100:.1f} %")
    metrics[2].metric("Contexts used", f"{stats.get('contexts_used', 0):,} of {stats.get('contexts_available', 0):,}")
    metrics[3].metric("Bits per bin", f"{result.payload_bytes * 8 / max(total_bins, 1):.3f}")
    st.caption(f"Binarization: {stats.get('mode', '')}. The header holds only shape and mode, because the model is learned on both sides.")

    frame = pd.DataFrame(cabac_rows)
    # Bypass bins have no state, so these columns must allow blanks without turning into floats.
    frame["state"] = frame["state"].astype("Int64")
    frame["MPS"] = frame["MPS"].astype("Int64")
    st.markdown("**Every bin of the first symbols**: binarization, chosen context, state and real cost")
    show_table(frame.drop(columns="context id").head(TABLE_ROWS).round({"p(LPS)": 4, "cost (bits)": 3, "total bits": 2}), height=340)

    if (frame["context id"] >= 0).any():
        show_chart(learning_figure(frame))
        st.caption(
            "Each context starts at p = 0.5 and walks the 64-state machine from H.264. p(LPS) cannot go below about 0.02, "
            "so a context-coded bin never costs less than about 0.03 bits. That is why static arithmetic coding wins on near-constant data."
        )

    st.markdown("**Contexts after coding the whole image** (busiest first)")
    show_table(pd.DataFrame(stats.get("context_table", [])).head(TABLE_ROWS).round({"share of 1": 4, "final p(LPS)": 4}), height=300)
    payload_preview(result)


# --- Benchmark --------------------------------------------------------------------

with tab_matrix:
    st.markdown("Runs every source against every predictor and coder, for the conclusions that no single image can give.")
    side = st.select_slider("Crop for this benchmark (px)", [64, 128, 256, 512], value=256)
    if st.button("Run benchmark", type="primary"):
        progress = st.progress(0.0, "Starting...")
        rows = benchmark.matrix_rows(
            sources.benchmark_sources(),
            lambda values, limit=side: sources.limit_size(values, limit),
            prediction.PREDICTORS,
            lambda fraction, text: progress.progress(fraction, text),
        )
        progress.empty()
        st.session_state["matrix"] = pd.DataFrame(rows)

    matrix = st.session_state.get("matrix")
    if matrix is not None:
        if not matrix["Lossless"].all():
            st.error("At least one run was not lossless.")
        else:
            st.success(f"All {len(matrix)} runs verified lossless.")

        chosen = st.radio("Prediction", list(prediction.PREDICTORS), index=2, horizontal=True, key="matrix_predictor")
        subset = matrix[matrix["Prediction"] == chosen]
        order = list(subset["Source"].drop_duplicates())
        figure = go.Figure()
        for name in CODERS:
            rows = subset[subset["Coder"] == name]
            figure.add_bar(
                y=rows["Source"],
                x=rows["Bits/pixel"],
                name=name,
                orientation="h",
                marker_color=COLORS[name],
                marker_line=dict(color=SURFACE, width=2),
                hovertemplate=name + ": %{x:.3f} bits/pixel<extra>%{y}</extra>",
            )
        figure.add_scatter(
            y=subset[subset["Coder"] == "Huffman"]["Source"],
            x=subset[subset["Coder"] == "Huffman"]["Entropy H0 (bits)"],
            name="Entropy H0",
            mode="markers",
            marker=dict(color="#0b0b0b", symbol="line-ns-open", size=14, line=dict(width=2)),
            hovertemplate="H0: %{x:.3f} bits/symbol<extra>%{y}</extra>",
        )
        figure.update_layout(barmode="group", bargap=0.25)
        figure.update_yaxes(categoryorder="array", categoryarray=order[::-1], automargin=True)
        show_chart(chart_layout(figure, f"Bits per pixel. Prediction: {chosen}", "Bits per pixel", "", height=64 + 44 * len(order)))
        st.caption("The black tick is the order-0 entropy of the symbols. Only a coder using context can pass it.")

        show_table(matrix.round({"Entropy H0 (bits)": 3, "Bits/pixel": 3, "Ratio": 3, "Encode s": 3, "Decode s": 3}))
        st.download_button("Download CSV", matrix.to_csv(index=False), "benchmark.csv", "text/csv")
