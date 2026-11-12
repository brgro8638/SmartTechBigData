import io
import os

import cv2
import numpy as np
import pydicom
import streamlit as st
from scipy.fft import dctn, idctn, fft2, ifft2, fftshift, ifftshift
from skimage.metrics import peak_signal_noise_ratio

# gjenbruker enkelte deler av implementasjonen fra HuffmanGUI (Smart Tech 1-2b)
from huffman import build_huffman_tree, generate_huffman_codes

st.set_page_config(page_title="DCT vs FFT codec", layout="wide")

# --- loading ---
def normalize(image):
    image = image.astype(np.float64)
    rng = image.max() - image.min()
    return np.zeros_like(image) if rng == 0 else (image - image.min()) / rng * 255

def loadDicom(data):
    image = pydicom.dcmread(io.BytesIO(data)).pixel_array.astype(np.float64)
    if image.ndim == 3:  # multi-frame: take first frame (assumes grayscale)
        image = image[0]
    return normalize(image)

def loadImage(data):
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise ValueError("Could not decode image")
    return normalize(decoded)

@st.cache_data(show_spinner=False)
def loadAny(data, filename):
    """Pick the load method from the file extension."""
    if os.path.splitext(filename)[1].lower() == ".dcm":
        return loadDicom(data)
    return loadImage(data)

# JPEG luminance quantization table (quality 50)
# https://www.sciencedirect.com/topics/computer-science/quantization-table
baseQuantTable = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
])

def byteHistogram(*arrays):
    values = np.concatenate([array.ravel() for array in arrays]).astype(np.int64)
    current = ((values << 1) ^ (values >> 63)).astype(np.uint64)  # zigzag
    histogram = np.zeros(256, dtype=np.int64)
    low7, seven = np.uint64(0x7F), np.uint64(7)
    while current.size:
        more = (current >> seven) > 0
        byteValues = (current & low7) | (more.astype(np.uint64) << seven)
        histogram += np.bincount(byteValues.astype(np.int64), minlength=256)
        current = current[more] >> seven
    return histogram

CODEBOOK_ENTRY_BYTES = 2

def byteHuffmanSizes(*arrays, halve=False):
    histogram = byteHistogram(*arrays)
    freqDict = {int(symbol): int(histogram[symbol]) for symbol in np.flatnonzero(histogram)}

    tree = build_huffman_tree(freqDict)
    codes = {}
    generate_huffman_codes(tree, "", codes)

    payloadBits = sum(freqDict[symbol] * max(len(code), 1) for symbol, code in codes.items())
    beforeBytes = float(histogram.sum())
    payloadBytes = payloadBits / 8

    # relevant for FFT conjugate symmetry
    if halve:
        beforeBytes /= 2
        payloadBytes /= 2
    return beforeBytes, payloadBytes + len(freqDict) * CODEBOOK_ENTRY_BYTES

def formatSize(numBytes):
    if numBytes >= 1024 ** 2:
        return f"{numBytes / 1024 ** 2:.2f} MiB"
    return f"{numBytes / 1024:.1f} KiB"

# Each codec returns (nonzeroCount, reconstruction, sizeBeforeHuffman, sizeAfterHuffman), sizes in bytes

# --- DCT codec: 8x8 blocks, JPEG table scaled by quality ---
@st.cache_data(show_spinner=False)
def dctCodec(image, quality=50):
    scaleFactor = 5000 / quality if quality < 50 else 200 - 2 * quality
    quantTable = np.clip((baseQuantTable * scaleFactor + 50) // 100, 1, 255)
    height, width = image.shape
    padded = np.pad(image, ((0, (-height) % 8), (0, (-width) % 8)), mode="edge")
    paddedHeight, paddedWidth = padded.shape
    blocks = padded.reshape(paddedHeight // 8, 8, paddedWidth // 8, 8).swapaxes(1, 2)  # (blockRows, blockCols, 8, 8)
    quantized = np.round(dctn(blocks - 128, axes=(2, 3), norm="ortho") / quantTable)
    reconBlocks = idctn(quantized * quantTable, axes=(2, 3), norm="ortho") + 128
    reconstruction = reconBlocks.swapaxes(1, 2).reshape(paddedHeight, paddedWidth)[:height, :width]
    return (np.count_nonzero(quantized), np.clip(reconstruction, 0, 255),
            *byteHuffmanSizes(quantized))

# --- FFT codec: whole image, step size grows with distance from DC ---
@st.cache_data(show_spinner=False)
def fftCodec(image, baseStep=2.0, growthLength=16.0):
    height, width = image.shape
    spectrum = fftshift(fft2(image - 128, norm="ortho"))
    rowIndices, colIndices = np.indices((height, width))
    distanceFromDc = np.hypot(rowIndices - height // 2, colIndices - width // 2)
    stepSize = baseStep * (1 + distanceFromDc / growthLength)
    quantizedReal = np.round(spectrum.real / stepSize)  # quantize real/imaginary separately
    quantizedImag = np.round(spectrum.imag / stepSize)
    reconstruction = ifft2(ifftshift((quantizedReal + 1j * quantizedImag) * stepSize), norm="ortho").real + 128
    # halve the count: conjugate symmetry means half the coefficients are redundant
    nonzeroCount = (np.count_nonzero(quantizedReal) + np.count_nonzero(quantizedImag)) // 2
    return (nonzeroCount, np.clip(reconstruction, 0, 255),
            *byteHuffmanSizes(quantizedReal, quantizedImag, halve=True))

# --- FFT codec: 8x8 blocks, step size grows with true frequency magnitude ---
@st.cache_data(show_spinner=False)
def fftBlockCodec(image, baseStep=8.0, growthLength=1.5):
    height, width = image.shape
    padded = np.pad(image, ((0, (-height) % 8), (0, (-width) % 8)), mode="edge")
    paddedHeight, paddedWidth = padded.shape
    blocks = padded.reshape(paddedHeight // 8, 8, paddedWidth // 8, 8).swapaxes(1, 2)

    # index k and 8-k are the same frequency, so fold the wraparound
    foldedFreq = np.minimum(np.arange(8), 8 - np.arange(8))
    freqMagnitude = np.hypot(foldedFreq[:, None], foldedFreq[None, :])
    stepSize = baseStep * (1 + freqMagnitude / growthLength)

    spectrum = fft2(blocks - 128, axes=(2, 3), norm="ortho")
    quantizedReal = np.round(spectrum.real / stepSize)
    quantizedImag = np.round(spectrum.imag / stepSize)
    reconBlocks = ifft2((quantizedReal + 1j * quantizedImag) * stepSize, axes=(2, 3), norm="ortho").real + 128
    reconstruction = reconBlocks.swapaxes(1, 2).reshape(paddedHeight, paddedWidth)[:height, :width]
    nonzeroCount = (np.count_nonzero(quantizedReal) + np.count_nonzero(quantizedImag)) // 2
    return (nonzeroCount, np.clip(reconstruction, 0, 255),
            *byteHuffmanSizes(quantizedReal, quantizedImag, halve=True))

toUint8 = lambda array: np.clip(array, 0, 255).astype(np.uint8)

# --- GUI ---
st.title("DCT vs FFT image codec comparison")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Load an image",
        type=["dcm", "png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )

    with st.form("codecParams"):
        st.header("DCT (8x8 blocks)")
        dctQuality = st.slider("Quality", 1, 100, 50)

        st.header("FFT (whole image)")
        fftBaseStep = st.slider("Base step", 0.1, 20.0, 2.0, 0.1, key="fftBase")
        fftGrowth = st.slider("Growth length", 1.0, 100.0, 16.0, 1.0, key="fftGrowth")

        st.header("FFT (8x8 blocks)")
        blockBaseStep = st.slider("Base step", 0.1, 40.0, 8.0, 0.1, key="blockBase")
        blockGrowth = st.slider("Growth length", 0.1, 10.0, 1.5, 0.1, key="blockGrowth")

        st.form_submit_button("Run compression", type="primary", width="stretch")

    showError = st.checkbox("Show error maps", value=False)

if uploaded is None:
    st.info("Upload a DICOM (.dcm) or regular image (.png, .jpg, ...) in the sidebar to get started.")
    st.stop()

try:
    originalImage = loadAny(uploaded.getvalue(), uploaded.name)
except Exception as error:
    st.error(f"Failed to load {uploaded.name}: {error}")
    st.stop()

isDicom = os.path.splitext(uploaded.name)[1].lower() == ".dcm"
st.caption(f"{uploaded.name} ({'DICOM' if isDicom else 'standard image'}), {originalImage.shape[1]}x{originalImage.shape[0]} px")

with st.spinner("Running codecs and Huffman coding..."):
    codecOutputs = [
        ("DCT", *dctCodec(originalImage, dctQuality)),
        ("FFT (whole image)", *fftCodec(originalImage, fftBaseStep, fftGrowth)),
        ("FFT (8x8 blocks)", *fftBlockCodec(originalImage, blockBaseStep, blockGrowth)),
    ]

# metrics table
rawBytes = originalImage.size  # 8-bit grayscale, 1 byte per pixel
st.caption(
    f"Uncompressed size (8-bit grayscale): {formatSize(rawBytes)} · uploaded file: {formatSize(uploaded.size)}. "
    "Huffman encoding is done at byte level."
)

rows = []
results = []  # (label, reconstruction, totalBytes) for the image captions below
for label, nonzeroCount, reconstruction, sizeBefore, totalBytes in codecOutputs:
    results.append((label, reconstruction, totalBytes))
    rows.append({
        "Codec": label,
        "PSNR (dB)": round(float(peak_signal_noise_ratio(originalImage, reconstruction, data_range=255)), 2),
        "Nonzero coeffs": int(nonzeroCount),
        "Nonzero fraction": f"{nonzeroCount / originalImage.size:.1%}",
        "Size before Huffman": formatSize(sizeBefore),
        "Compressed size (Huffman)": formatSize(totalBytes),
        "Compression ratio": f"{rawBytes / totalBytes:.1f}:1" if totalBytes > 0 else "n/a",
        "Size vs. raw": f"{totalBytes / rawBytes:.1%}",
    })
st.dataframe(rows, hide_index=True, width="stretch")

# images: original | DCT | FFT whole | FFT blocks
columns = st.columns(4)
columns[0].image(toUint8(originalImage), caption="Original", width="stretch")
for column, (label, reconstruction, totalBytes) in zip(columns[1:], results):
    column.image(toUint8(reconstruction), caption=f"{label} (~{formatSize(totalBytes)})", width="stretch")

if showError:
    st.subheader("Absolute error (amplified 4x)")
    errorColumns = st.columns(3)
    for column, (label, reconstruction, _) in zip(errorColumns, results):
        column.image(toUint8(np.abs(originalImage - reconstruction) * 4), caption=label, width="stretch")