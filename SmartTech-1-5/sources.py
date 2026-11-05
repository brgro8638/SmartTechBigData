"""Data sources: generated images, Harnverhalt DICOM, real photos and raw files.

Every loader returns a Source: a 2D integer array plus the bit depth it nominally
occupies, which is the baseline for the compression ratio.
"""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "samples"
HARNVERHALT_NAME = "Harnverhalt2.zip"


@dataclass
class Source:
    name: str
    category: str
    pixels: np.ndarray
    nominal_bits: int
    note: str = ""


def center_crop(pixels: np.ndarray, max_side: int) -> np.ndarray:
    """Crop rather than downscale: resampling would change the noise statistics."""
    height, width = pixels.shape
    new_height, new_width = min(height, max_side), min(width, max_side)
    top = (height - new_height) // 2
    left = (width - new_width) // 2
    return pixels[top:top + new_height, left:left + new_width]


def limit_size(pixels: np.ndarray, max_side: int) -> np.ndarray:
    """A single-row stream keeps the same sample budget as a square image would."""
    if pixels.shape[0] == 1:
        return pixels[:, : max_side * max_side]
    return center_crop(pixels, max_side)


# --- Generated data -------------------------------------------------------------

def _gradient(size: int, rng: np.random.Generator) -> np.ndarray:
    ramp = np.linspace(0, 255, size)
    return np.round((ramp[None, :] + ramp[:, None]) / 2).astype(np.int64)


def _checkerboard(size: int, rng: np.random.Generator) -> np.ndarray:
    tile = max(1, size // 16)
    grid = np.indices((size, size)) // tile
    return ((grid[0] + grid[1]) % 2 * 255).astype(np.int64)


def _uniform_noise(size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, (size, size), dtype=np.int64)


def _gaussian_noise(size: int, rng: np.random.Generator) -> np.ndarray:
    return np.clip(np.round(rng.normal(128, 12, (size, size))), 0, 255).astype(np.int64)


def _skewed_binary(size: int, rng: np.random.Generator) -> np.ndarray:
    return (rng.random((size, size)) < 0.02).astype(np.int64)


def _constant(size: int, rng: np.random.Generator) -> np.ndarray:
    return np.full((size, size), 128, dtype=np.int64)


def _shapes(size: int, rng: np.random.Generator) -> np.ndarray:
    """Flat regions with sharp edges and mild noise, like a simple phantom."""
    rows, columns = np.indices((size, size)) / size
    image = np.full((size, size), 40.0)
    image[(rows - 0.5) ** 2 / 0.16 + (columns - 0.5) ** 2 / 0.10 < 1] = 120
    image[(rows - 0.38) ** 2 + (columns - 0.40) ** 2 < 0.010] = 220
    image[(rows - 0.62) ** 2 + (columns - 0.60) ** 2 < 0.018] = 180
    image[(rows > 0.72) & (rows < 0.78) & (columns > 0.3) & (columns < 0.7)] = 250
    return np.clip(np.round(image + rng.normal(0, 2, image.shape)), 0, 255).astype(np.int64)


GENERATORS = {
    "Shapes with noise": _shapes,
    "Smooth gradient": _gradient,
    "Checkerboard": _checkerboard,
    "Gaussian noise": _gaussian_noise,
    "Uniform noise": _uniform_noise,
    "Skewed binary (2 % ones)": _skewed_binary,
    "Constant": _constant,
}


def generated(name: str, size: int, seed: int) -> Source:
    pixels = GENERATORS[name](size, np.random.default_rng(seed))
    return Source(name, "Generated", pixels, 8)


# --- Medical data ---------------------------------------------------------------

def find_harnverhalt_zip() -> Path | None:
    """Look in data/, then walk up: the zip is gitignored and lives at the repo root."""
    configured = os.environ.get("HARNVERHALT_ZIP")
    if configured and Path(configured).is_file():
        return Path(configured)
    for folder in [HERE / "data", HERE, *HERE.parents]:
        candidate = folder / HARNVERHALT_NAME
        if candidate.is_file():
            return candidate
    return None


def list_dicom_members(archive_bytes: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        return sorted(name for name in archive.namelist() if name.lower().endswith(".dcm"))


def read_dicom_member(archive_bytes: bytes, member: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        return archive.read(member)


def dicom_source(dicom_bytes: bytes, label: str, representation: str,
                 center: float = 40, width: float = 400) -> Source:
    """Decode pixels only. Patient metadata is never read into the app.

    "stored": the values the scanner stored (the real clinical data).
    "windowed": soft-tissue window mapped to 8 bits, i.e. what is shown on screen.
    """
    import pydicom

    dataset = pydicom.dcmread(io.BytesIO(dicom_bytes))
    stored = dataset.pixel_array.astype(np.int64)
    if stored.ndim != 2:
        raise ValueError(f"Expected a single 2D slice, got shape {stored.shape}")
    file_note = f"The DICOM file itself holds this slice in {len(dicom_bytes) / 1024:.0f} KiB"
    syntax = dataset.file_meta.TransferSyntaxUID
    if syntax.is_compressed:
        file_note += f" ({syntax.name.split(',')[0]})."
    else:
        file_note += " (uncompressed)."

    if representation == "stored":
        return Source(label, "Medical", stored, int(dataset.BitsStored), file_note)

    slope = float(getattr(dataset, "RescaleSlope", 1))
    intercept = float(getattr(dataset, "RescaleIntercept", 0))
    hounsfield = stored * slope + intercept
    low = center - width / 2
    windowed = np.clip((hounsfield - low) / width, 0, 1) * 255
    file_note += " Windowing clips everything outside the window, so only the stored values are the full data."
    return Source(label, "Medical", np.round(windowed).astype(np.int64), 8, file_note)


# --- Real-life photos -----------------------------------------------------------

def list_samples() -> list[str]:
    if not SAMPLES.is_dir():
        return []
    return sorted(path.name for path in SAMPLES.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})


def photo_source(image_bytes: bytes, label: str) -> Source:
    """Photos are compared on luma: one plane keeps all coders on the same footing."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        gray = image.convert("L")
        return Source(label, "Real-life", np.asarray(gray, dtype=np.int64), 8)


def sample_photo(name: str) -> Source:
    return photo_source((SAMPLES / name).read_bytes(), name)


# --- Raw files ------------------------------------------------------------------

def raw_source(data: bytes, label: str, sample_bits: int, width: int, big_endian: bool = False) -> Source:
    """Interpret any file as unsigned samples. Width 0 means one long row."""
    if sample_bits == 8:
        values = np.frombuffer(data, dtype=np.uint8)
    elif sample_bits == 16:
        values = np.frombuffer(data[: len(data) // 2 * 2], dtype=">u2" if big_endian else "<u2")
    else:
        raise ValueError("Raw samples must be 8 or 16 bits")
    if values.size == 0:
        raise ValueError("The raw file is empty")

    if width <= 0:
        width = values.size
    height = values.size // width
    if height == 0:
        raise ValueError(f"Width {width} is larger than the {values.size} samples in the file")
    note = ""
    dropped = values.size - height * width
    if dropped:
        note = f"{dropped} trailing samples do not fill a row and were ignored."
    pixels = values[: height * width].reshape(height, width).astype(np.int64)
    return Source(label, "Raw file", pixels, sample_bits, note)


# --- The fixed set behind the benchmark tab ---------------------------------------

def benchmark_sources() -> list[Source]:
    """Every generated pattern and sample photo, plus one mid-series CT slice two ways."""
    items = [generated(name, 512, 1) for name in GENERATORS]
    items += [sample_photo(name) for name in list_samples()]
    path = find_harnverhalt_zip()
    if path is not None:
        archive = path.read_bytes()
        members = list_dicom_members(archive)
        data = read_dicom_member(archive, members[len(members) // 2])
        for representation, label in (("windowed", "window 8 bit"), ("stored", "stored 12 bit")):
            item = dicom_source(data, f"Harnverhalt CT, {label}", representation)
            items.append(item)
    return items
