import struct
import zlib

import cv2
import numpy as np
import pywt
import streamlit as st

MAGIC = b"WVLT" # Kan være hva som helst, så lenge den er lik ved embed og extract. Må være 4 byte, ellers må vi endre header-format.
HEADER_FMT = ">4sII" # ">" angir big-endian, "4s" angir 4 byte string "magic" (se over), "I" er uint32, vi har en for lengden på data og en for crc32.
HEADER_BITS = struct.calcsize(HEADER_FMT) * 8
WAVELETS = ["haar", "db2", "db4", "sym4", "coif2"]

###
# Kjerne-metoder for wavelet
###
def _get_layer(img, wavelet):
    height, width = img.shape[:2]
    height2, width2 = height - height % 2, width - width % 2      
    channel = img[:height2, :width2, 0].astype(np.float64)
    # cA = low-low, cH = high-low, cV = low-high, cD = high-high
    cA, (cH, cV, cD) = pywt.dwt2(channel, wavelet, mode="periodization")
    layer = np.stack([cH, cV, cD]).ravel()
    return cA, layer, cH.shape, (height2, width2)

def _positions(size, key):
    return np.random.default_rng(key).permutation(size)

def embed(img, message, wavelet, amp, key):
    cA, layer, dshape, (h2, w2) = _get_layer(img, wavelet)

    header = struct.pack(HEADER_FMT, MAGIC, len(message), zlib.crc32(message))
    bits = np.unpackbits(np.frombuffer(header + message, dtype=np.uint8))
    if bits.size > layer.size:
        capacity = (layer.size - HEADER_BITS) // 8
        raise ValueError(f"Message too long: this image holds at most {capacity} bytes.")

    # tildeldig spredning for jevn fordeling av data (tydelig visualisert i differens-bildet)
    idx = _positions(layer.size, key)[: bits.size]
    # bytter ut layer
    layer[idx] = np.where(bits == 1, amp, -amp) # 1 -> amp, 0 -> -amp    

    cH, cV, cD = layer.reshape(3, *dshape)
    rec = pywt.idwt2((cA, (cH, cV, cD)), wavelet, mode="periodization")

    out = img.copy()
    out[:h2, :w2, 0] = np.clip(np.rint(rec), 0, 255).astype(np.uint8)
    return out

def extract(img, wavelet, key):
    _, layer, _, _ = _get_layer(img, wavelet)
    if layer.size < HEADER_BITS:
        return None
    order = _positions(layer.size, key)

    def read(start, nbits):
        return (layer[order[start:start + nbits]] > 0).astype(np.uint8)

    header = np.packbits(read(0, HEADER_BITS)).tobytes()
    magic, length, crc = struct.unpack(HEADER_FMT, header)
    if magic != MAGIC or HEADER_BITS + length * 8 > layer.size:
        return None
    data = np.packbits(read(HEADER_BITS, length * 8)).tobytes()
    return data if zlib.crc32(data) == crc else None

###
# Metoder som benyttes direkte av GUI
###
def bytes_to_img(data):
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

def decode_upload(upload):
    return bytes_to_img(upload.getvalue())

def encode_png(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Couldn't encode the result as PNG.")
    return buf.tobytes()

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def capacity_bytes(img):
    h, w = img.shape[:2]
    coeffs = 3 * ((h - h % 2) // 2) * ((w - w % 2) // 2)
    return max((coeffs - HEADER_BITS) // 8, 0)

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))

def difference_image(a, b):
    """Blue-channel difference, contrast-stretched so it is visible."""
    diff = np.abs(a[:, :, 0].astype(np.int16) - b[:, :, 0].astype(np.int16)).astype(np.uint8)
    peak = int(diff.max())
    if peak == 0:
        return diff, 0
    return (diff.astype(np.float64) * (255.0 / peak)).astype(np.uint8), peak

def run_embed(cover, payload, wavelet, amp, key):
    # legg inn hemmelig data, og encode resultatet som png (lossless)
    stego = embed(cover, payload, wavelet, amp, key)
    png = encode_png(stego)
    # kontroller round-trip (data hentet ut av png verifiseres mot checksum med zlib)
    verified = extract(bytes_to_img(png), wavelet, key) == payload
    # differens-bilde for illustrasjon av data-fordeling + største individuelle pixel-endring (mest fordi det er interessant)
    diff, peak = difference_image(cover, stego)
    return {
        "cover": cover,
        "stego": stego,
        "png": png,
        "verified": verified,
        "diff": diff,
        "peak": peak,
        "psnr": psnr(cover, stego),
        "size": len(payload),
        "capacity": capacity_bytes(cover),
    }

###
# Streamlit GUI
###
# Generelt side-oppsett
st.set_page_config(page_title="Wavelet steganography", layout="wide")

st.title("Wavelet steganography")
st.caption("Hide a message in an image by replacing one wavelet detail layer.")

with st.sidebar:
    st.header("Settings")
    wavelet = st.selectbox(
        "Wavelet", WAVELETS,
        help="Wavelet family used for the transform. Extraction must use the same one.",
    )
    amp = st.slider(
        "Amplitude", 1.0, 30.0, 6.0, 0.5,
        help="Value written into the detail layer. Higher survives rounding better "
             "but changes the image more.",
    )
    key = int(st.number_input(
        "Key", min_value=0, value=42, step=1,
        help="Seed for the position shuffle. Extraction must use the same key.",
    ))

embed_tab, extract_tab = st.tabs(["Embed", "Extract"])

# Embed / Legge data inn i et bilde
with embed_tab:
    col_in, col_prev = st.columns(2, gap="large")

    with col_in:
        cover_file = st.file_uploader(
            "Cover image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="cover",
        )
        kind = st.radio("Payload", ["Text", "File"], horizontal=True)
        if kind == "Text":
            text = st.text_area(
                "Message", height=120,
                placeholder="Meta information to hide, e.g. author and date",
            )
            payload = text.encode("utf-8") if text else b""
        else:
            payload_file = st.file_uploader("File to hide", key="payload")
            payload = payload_file.getvalue() if payload_file else b""
        run = st.button("Embed message", type="primary")

    cover = decode_upload(cover_file) if cover_file else None
    too_big = cover is not None and len(payload) > capacity_bytes(cover)
    
    if too_big:
        st.warning(
            f"The payload is {len(payload):,} bytes, but this image holds at most "
            f"{capacity_bytes(cover):,}. Shorten the message, pick a smaller file, "
            "or use a larger image."
        )

    with col_prev:
        if cover_file and cover is None:
            st.error("Couldn't read that image. Try a PNG or JPEG.")
        elif cover is not None:
            st.image(to_rgb(cover), caption="Cover image")
            st.caption(
                f"Capacity: {capacity_bytes(cover):,} bytes. "
                f"Payload: {len(payload):,} bytes."
            )

    if run:
        if cover is None:
            st.warning("Upload a cover image first.")
        elif not payload:
            st.warning("Enter a message or choose a file to hide.")
        else:
            try:
                with st.spinner("Embedding…"):
                    st.session_state["result"] = run_embed(cover, payload, wavelet, amp, key)
            except ValueError as e:
                st.session_state.pop("result", None)
                st.error(str(e))

    res = st.session_state.get("result")
    if res:
        st.divider()
        if res["verified"]:
            st.success(f"Embedded {res['size']:,} bytes and verified the round trip.")
        else:
            st.error(
                "The image was saved, but the message didn't survive a round trip. "
                "Try a larger amplitude or a less saturated image."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PSNR", "inf" if np.isinf(res["psnr"]) else f"{res['psnr']:.1f} dB")
        m2.metric("Max pixel change", f"{res['peak']}")
        m3.metric("Payload", f"{res['size']:,} bytes")
        m4.metric("Capacity used", f"{100 * res['size'] / max(res['capacity'], 1):.2f}%")

        c1, c2, c3 = st.columns(3)
        c1.image(to_rgb(res["cover"]), caption="Original")
        c2.image(to_rgb(res["stego"]), caption="With hidden message")
        c3.image(res["diff"], caption="Difference, blue channel (contrast-stretched)")

        st.download_button(
            "Download PNG", res["png"], file_name="stego.png", mime="image/png",
        )

# Extract / Hente data ut av et bilde
with extract_tab:
    stego_file = st.file_uploader(
        "Image with a hidden message",
        type=["png"], key="stego_in",
    )
    if stego_file is not None:
        
        stego = decode_upload(stego_file)
        if stego is None:
            st.error("Couldn't read that image.")
        else:
            left, right = st.columns(2, gap="large")
            with left:
                st.image(to_rgb(stego), caption="Uploaded image")
            with right:
                data = extract(stego, wavelet, key)
                if data is None:
                    st.error(
                        "No message found. Check that the wavelet and key match the ones "
                        "used to embed, and that the image wasn't resized, compressed or edited."
                    )
                else:
                    st.success(f"Found a hidden message ({len(data):,} bytes).")
                    try:
                        st.text_area("Message", value=data.decode("utf-8"), height=160)
                    except UnicodeDecodeError:
                        st.info("The hidden data is binary, not text.")
                        st.download_button(
                            "Download hidden file", data, file_name="hidden_message.bin",
                        )
