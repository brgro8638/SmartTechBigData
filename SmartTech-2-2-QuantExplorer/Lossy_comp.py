import streamlit as st
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

st.title("Lossy Compression using DCT")
st.write("Compare three different compression scenarios.")

matrix_a = np.zeros((8, 8))
matrix_a[:4, :4] = 1

matrix_b = np.ones((8, 8))
matrix_b[:4, :4] = 0

matrix_c = np.zeros((8, 8))
matrix_c[2:6, 2:6] = 1

def compress(image, matrix):
    height, width = image.shape
    result = np.zeros((height, width))

    for y in range(0, height, 8):
        for x in range(0, width, 8):
            block = image[y:y+8, x:x+8]

            dct_block = dctn(block - 128, norm="ortho")

            quantized = np.round(dct_block / 10) * matrix

            restored = quantized * 10

            result[y:y+8, x:x+8] = (
                idctn(restored, norm="ortho") + 128
            )

    return np.clip(result, 0, 255).astype(np.uint8)

def calculate_psnr(original, compressed):
    mse = np.mean(
        (original.astype(float) - compressed.astype(float)) ** 2
    )

    if mse == 0:
        return float("inf")

    return 10 * np.log10((255 ** 2) / mse)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    image = image.resize((256, 256))
    original = np.array(image)

    st.subheader("Original image")
    st.image(image, width=256)

    scenarios = {
        "A: Normal compression": matrix_a,
        "B: Remove low frequencies": matrix_b,
        "C: Keep middle frequencies": matrix_c
    }

    for name, matrix in scenarios.items():

        compressed = compress(original, matrix)
        psnr = calculate_psnr(original, compressed)

        st.subheader(name)

        col1, col2 = st.columns(2)

        with col1:
            st.write("Compressed image")
            st.image(compressed, width=256)

        st.write(f"PSNR: {psnr:.2f} dB")
        st.divider()