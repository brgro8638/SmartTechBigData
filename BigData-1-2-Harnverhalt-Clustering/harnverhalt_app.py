import glob
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_PATH = "/Users/bredegronningen/Documents/Datateklokal/Big Data:Smart Tech/Lectures/Harnverhalt"


# Streamlit kjører hele skriptet på nytt ved hver endring, derfor caches tunge lesinger.
@st.cache_data
def last_hu(path):
    """Les DICOM-fila: sti inn -> HU-array ut."""
    if isinstance(path, (bytes, bytearray)):
        # TODO: finnes ikke i notebooken. Bytes gjør at opplastede og lokale filer bruker samme funksjon.
        ds = pydicom.dcmread(io.BytesIO(path))
    else:
        ds = pydicom.dcmread(path)
    if "PixelData" not in ds:
        raise ValueError(f"Fila manglar Pixel Data: {path}")

    raw = ds.pixel_array

    # Bruk metadata dersom fila har Hounsfield-konvertering.
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    hu = raw.astype(np.float32) * slope + intercept
    return hu


def window(hu, center=40, width=400):
    """Vindusbehandle HU-bildet til 0–255."""
    low = center - width / 2
    high = center + width / 2
    clipped = np.clip(hu, low, high)
    scaled = (clipped - low) / (high - low) * 255
    return scaled.astype(np.uint8)


def bladder_mask(features, rad, kol, k=5):
    """features: tabell (antall piksler, antall kolonner). Returnerer en True/False-maske for blæreklyngen."""
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    seg = km.fit_predict(features).reshape(b.shape)   # klynge, og gjør om til bilde igjen
    label = seg[rad, kol]                             # hvilken klynge blærepikselen havnet i
    return seg == label                               # maske: True der klyngen er


def dice(a, b):
    """Dice-koeffisient: overlapp mellom to masker, fra 0 til 1."""
    denominator = a.sum() + b.sum()
    return 2 * (a & b).sum() / denominator if denominator else 1.0


@st.cache_data
def kmeans_segmentering(features, shape, k=5):
    """K-means fra notebooken, med cache fordi beregningen er tung."""
    # Streamlit kjører hele skriptet på nytt ved hver endring, derfor caches k-means-resultatet.
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    seg = km.fit_predict(features).reshape(shape)
    return seg, km.cluster_centers_


@st.cache_data
def lag_pca_features(Xs):
    """TODO: finnes ikke i notebooken. Cacher notebookens PCA-kall til segmenteringsfanen."""
    pca = PCA(n_components=2)
    return pca.fit_transform(Xs), pca.explained_variance_ratio_


@st.cache_data
def pca_rekonstruksjon(bilde, n):
    """Komprimer bildet til n komponenter med PCA og bygg det opp igjen."""
    pca = PCA(n_components=n)
    Z = pca.fit_transform(bilde)        # (512, n): hver bilderad beskrevet med n tall
    rek = pca.inverse_transform(Z)      # (512, 512): bildet bygget opp igjen
    return np.clip(rek, 0, 255), pca.explained_variance_ratio_.sum()


def psnr(original, rek):
    """Peak Signal-to-Noise Ratio i dB. Høyere er bedre. Over ~30 dB er tapet knapt synlig."""
    mse = np.mean((original - rek) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255 ** 2 / mse)


def morfologi(original, operasjon, kernel_size=5):
    """TODO: finnes ikke i notebooken. Samler notebookens morfologioperasjoner for UI-et."""
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    if operasjon == "erosjon":
        return cv2.erode(original, k)
    if operasjon == "dilasjon":
        return cv2.dilate(original, k)
    if operasjon == "opening":
        return cv2.morphologyEx(original, cv2.MORPH_OPEN, k)
    if operasjon == "closing":
        return cv2.morphologyEx(original, cv2.MORPH_CLOSE, k)
    if operasjon == "opening+closing":
        open_bilde = cv2.morphologyEx(original, cv2.MORPH_OPEN, k)
        return cv2.morphologyEx(open_bilde, cv2.MORPH_CLOSE, k)
    return original


def lag_features(b):
    """TODO: finnes ikke i notebooken. Gjør feature-cellen gjenbrukbar i appen."""
    snitt = cv2.blur(b, (7, 7))
    std = np.sqrt(np.maximum(cv2.blur(b**2, (7, 7)) - snitt**2, 0))

    gx = cv2.Sobel(b, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(b, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx**2 + gy**2)

    F = np.stack([b, snitt, std, grad], axis=-1)
    return F, F.reshape(-1, 4)


def filer_fra_kilde(mappe, opplastet):
    """TODO: finnes ikke i notebooken. Velger mappefiler eller én opplastet fil."""
    if opplastet is not None:
        return [(opplastet.name, opplastet.getvalue())]
    return [
        (os.path.basename(sti), sti)
        for sti in sorted(glob.glob(os.path.join(mappe, "*.dcm")))
    ]


def vis_bilde(bilde, tittel, cmap="gray", colorbar=False, marker=None):
    """TODO: finnes ikke i notebooken. Viser alle bilder med matplotlib i appen."""
    fig, ax = plt.subplots()
    bildevisning = ax.imshow(bilde, cmap=cmap)
    if marker is not None:
        kol, rad = marker
        ax.plot(kol, rad, "r+", markersize=14, markeredgewidth=2)
    ax.set_title(tittel)
    ax.axis("off")
    if colorbar:
        fig.colorbar(bildevisning, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)


def areal_cm2(mask, kilde):
    """TODO: finnes ikke i notebooken. Regner maskeareal fra DICOM-metadata."""
    ds = pydicom.dcmread(io.BytesIO(kilde) if isinstance(kilde, bytes) else kilde)
    pixel_spacing = getattr(ds, "PixelSpacing", [1, 1])
    return float(mask.sum()) * float(pixel_spacing[0]) * float(pixel_spacing[1]) / 100


st.set_page_config(page_title="Harnverhalt CT-analyse", layout="wide")
st.title("Analyse av CT-bilder av urinblære")

with st.sidebar:
    st.header("Datavalg")
    mappe = st.text_input("Mappe med .dcm-filer", DEFAULT_PATH)
    opplastet = st.file_uploader("Last opp én .dcm-fil", type=["dcm"])
    kildefiler = filer_fra_kilde(mappe, opplastet)
    if not kildefiler:
        st.warning("Fant ingen .dcm-filer i mappen.")
        st.stop()

    første_hu = last_hu(kildefiler[0][1])
    snitt = st.slider("Snitt", 0, len(kildefiler) - 1, min(260, len(kildefiler) - 1))
    center = st.number_input("HU-senter", value=40, step=1)
    width = st.number_input("HU-bredde", min_value=1, value=400, step=1)
    operasjon = st.selectbox(
        "Morfologi",
        ["erosjon", "dilasjon", "opening", "closing", "opening+closing", "ingen"],
        index=2,
    )
    kernel_size = st.number_input("Kjernestørrelse", min_value=1, value=5, step=2)

filnavn, kilde = kildefiler[snitt]
hu = last_hu(kilde)
bilde = window(hu, center, width)
behandlet = morfologi(bilde, operasjon, int(kernel_size))
diff_morfologi = cv2.absdiff(bilde, behandlet)

segmentering_fane, rekonstruksjon_fane = st.tabs(
    ["Segmentering", "PCA-rekonstruksjon"]
)

with segmentering_fane:
    st.markdown(
        "Denne fanen viser hvordan HU-vinduet og morfologien påvirker bildet, "
        "og hvordan k-means deler pikslene i klynger."
    )

    venstre, høyre = st.columns(2)
    with venstre:
        vis_bilde(bilde, f"Vindusbehandlet original · {filnavn}")
    with høyre:
        visning = st.radio(
            "Vis steg",
            [
                "Etter morfologi",
                "Differanse (morfologi)",
                "K-means-segmenter",
                "Blæremaske",
            ],
            horizontal=True,
        )

    with st.container():
        km_k, km_input, km_rad, km_kol = st.columns([1, 2, 1, 1])
        with km_k:
            k = st.slider("k", 2, 8, 5)
        with km_input:
            input_type = st.radio(
                "K-means-input",
                ["1 feature", "4 features (standardisert)", "PCA (2 komponenter)"],
                horizontal=True,
            )
        with km_rad:
            rad = st.number_input("Referanse rad", min_value=0, value=225, step=1)
        with km_kol:
            kol = st.number_input("Referanse kolonne", min_value=0, value=250, step=1)

    if rad >= behandlet.shape[0] or kol >= behandlet.shape[1]:
        st.error("Referansepikselen ligger utenfor bildet.")
        st.stop()

    b = behandlet.astype(np.float32)
    _, X = lag_features(b)
    Xs = StandardScaler().fit_transform(X)
    pca_input, pca_varians = lag_pca_features(Xs)

    if input_type == "1 feature":
        kmeans_input = b.reshape(-1, 1)
    elif input_type == "4 features (standardisert)":
        kmeans_input = Xs
    else:
        kmeans_input = pca_input

    seg, sentre = kmeans_segmentering(kmeans_input, b.shape, k)
    mask = seg == seg[rad, kol]
    sammenlign = st.checkbox("Sammenlign med 1 feature")
    mask_1_feature = None
    seg_1_feature = None
    if sammenlign:
        seg_1_feature, _ = kmeans_segmentering(b.reshape(-1, 1), b.shape, k)
        mask_1_feature = seg_1_feature == seg_1_feature[rad, kol]

    with høyre:
        if visning == "Etter morfologi":
            vis_bilde(behandlet, "Etter morfologi")
        elif visning == "Differanse (morfologi)":
            vis_bilde(diff_morfologi, "Differanse (morfologi)", cmap="hot")
        elif visning == "K-means-segmenter":
            vis_bilde(seg, "K-means-segmenter", cmap="tab10", marker=(kol, rad))
        else:
            vis_bilde(mask, "Blæremaske", marker=(kol, rad))

    if sammenlign:
        sammenligning_venstre, sammenligning_høyre = st.columns(2)
        with sammenligning_venstre:
            vis_bilde(seg, f"Valgt input: {input_type}", cmap="tab10")
        with sammenligning_høyre:
            vis_bilde(seg_1_feature, "1 feature", cmap="tab10")
        st.metric("Dice mellom blæremaskene", f"{dice(mask, mask_1_feature):.3f}")

    metric_venstre, metric_høyre = st.columns(2)
    with metric_venstre:
        st.metric("Pikseltall i blæremaske", f"{int(mask.sum()):,}")
    with metric_høyre:
        st.metric("Areal", f"{areal_cm2(mask, kilde):.2f} cm²")

    if input_type == "PCA (2 komponenter)":
        st.metric(
            "Forklart varians PC1 + PC2",
            f"{(pca_varians[0] + pca_varians[1]):.1%}",
        )
    if input_type == "1 feature":
        hu_sentre = center - width / 2 + sentre.ravel() / 255 * width
        st.write(
            "Klyngesentre",
            {
                "Gråverdi (0–255)": sentre.ravel(),
                "HU": hu_sentre,
            },
        )

with rekonstruksjon_fane:
    st.markdown(
        "Denne fanen viser hvor mye av bildet som kan bevares med færre PCA-komponenter. "
        "PSNR måler bildekvalitet, mens lagringsandelen viser komprimeringskostnaden."
    )

    max_n = min(256, behandlet.shape[0], behandlet.shape[1])
    n = st.slider("Antall komponenter n", 1, max_n, min(50, max_n))
    rek, var = pca_rekonstruksjon(b, n)
    diff = np.abs(b - rek)
    psnr_verdi = psnr(b, rek)
    lagring = (b.shape[0] * n + n * b.shape[1] + b.shape[1]) / b.size

    metric_psnr, metric_var, metric_lagring = st.columns(3)
    metric_psnr.metric("PSNR", f"{psnr_verdi:.1f} dB")
    metric_var.metric("Forklart varians", f"{var:.1%}")
    metric_lagring.metric("Lagring av original", f"{lagring:.1%}")

    bildekolonner = st.columns(3)
    with bildekolonner[0]:
        vis_bilde(b, "Original (etter morfologi)")
    with bildekolonner[1]:
        vis_bilde(rek, f"Rekonstruksjon · n={n}")
    with bildekolonner[2]:
        vis_bilde(diff, "Differansebilde", cmap="hot", colorbar=True)

    ns = [verdi for verdi in [5, 10, 20, 50, 100, 150, 200, 255] if verdi <= max_n]
    resultater = []
    for n_verdi in ns:
        rek_n, var_n = pca_rekonstruksjon(b, n_verdi)
        lagring_n = (
            b.shape[0] * n_verdi + n_verdi * b.shape[1] + b.shape[1]
        ) / b.size
        resultater.append((n_verdi, psnr(b, rek_n), var_n, lagring_n))

    fig, ax_psnr = plt.subplots()
    ax_lagring = ax_psnr.twinx()
    ax_psnr.plot(
        [resultat[0] for resultat in resultater],
        [resultat[1] for resultat in resultater],
        "o-",
        label="PSNR (dB)",
    )
    ax_lagring.plot(
        [resultat[0] for resultat in resultater],
        [resultat[3] for resultat in resultater],
        "s-",
        color="tab:red",
        label="Lagringsandel",
    )
    ax_psnr.set_xlabel("Antall komponenter n")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_lagring.set_ylabel("Lagringsandel")
    ax_lagring.axvline(
        256,
        color="black",
        linestyle="--",
        label="samme lagring som original",
    )
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    st.pyplot(fig, clear_figure=True)
