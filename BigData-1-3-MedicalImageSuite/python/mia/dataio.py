"""Loading datasets (DICOM, .npy, images) and writing the shared result format."""

import io
import json
import zipfile
from collections import Counter

import numpy as np
from PIL import Image


def to_volume(array):
    """Any 2D/3D array -> float32 volume (slices, rows, cols). Same rules as toVolume() in JS."""
    array = np.asarray(array)
    if array.ndim == 2:
        return array[None].astype(np.float32)
    if array.ndim == 3 and array.shape[2] in (3, 4) and array.shape[0] > 4:
        rgb = array[..., :3].astype(np.float64)
        return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[None].astype(np.float32)
    if array.ndim == 3:
        return array.astype(np.float32)
    raise ValueError(f"Expected a 2D image or 3D volume, got shape {array.shape}")


def load_npy(data):
    return to_volume(np.load(io.BytesIO(data), allow_pickle=False))


def load_image(data):
    return to_volume(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")))


# ---------------------------------------------------------------------------
# DICOM (single file or zip with one or more series)
# ---------------------------------------------------------------------------

def _dicom_datasets(name, data):
    import pydicom
    files = []
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            files = [archive.read(n) for n in archive.namelist() if not n.endswith("/") and "__MACOSX" not in n]
    else:
        files = [data]
    datasets = []
    for raw in files:
        try:
            # force=True: many exported files lack the 128-byte preamble
            ds = pydicom.dcmread(io.BytesIO(raw), force=True)
        except Exception:
            continue
        if "PixelData" in ds:
            datasets.append(ds)
    if not datasets:
        raise ValueError("No DICOM images with pixel data found.")
    return datasets


def _slice_position(ds):
    """Position along the slice normal (works for axial, coronal and sagittal stacks)."""
    if "ImagePositionPatient" in ds and "ImageOrientationPatient" in ds:
        o = np.array(ds.ImageOrientationPatient, dtype=float)
        return float(np.dot(np.array(ds.ImagePositionPatient, dtype=float), np.cross(o[:3], o[3:])))
    return float(getattr(ds, "InstanceNumber", 0))


def dicom_series(name, data):
    """Group the images by series. Returns a list of dicts, largest series first."""
    groups = {}
    for ds in _dicom_datasets(name, data):
        groups.setdefault(getattr(ds, "SeriesInstanceUID", "unknown"), []).append(ds)
    series = []
    for items in groups.values():
        first = items[0]
        series.append({
            "label": f"{getattr(first, 'Modality', '?')} – {getattr(first, 'SeriesDescription', '') or 'no description'}"
                     f" ({len(items)} images, {first.Rows}×{first.Columns})",
            "datasets": items,
        })
    series.sort(key=lambda s: len(s["datasets"]), reverse=True)
    return series


def dicom_volume(datasets):
    """Stack one series into a float32 volume (HU for CT via the rescale tags)."""
    def rescale(ds):
        return ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) \
            + float(getattr(ds, "RescaleIntercept", 0))

    if len(datasets) == 1:
        return to_volume(rescale(datasets[0]))
    main_shape = Counter((int(d.Rows), int(d.Columns)) for d in datasets).most_common(1)[0][0]
    datasets = sorted((d for d in datasets if (int(d.Rows), int(d.Columns)) == main_shape), key=_slice_position)
    return np.stack([rescale(d) for d in datasets]).astype(np.float32)


# ---------------------------------------------------------------------------
# Shared result format
# ---------------------------------------------------------------------------

def npy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def result_json(config, result, implementation, dataset_name):
    return json.dumps({
        "implementation": implementation,
        "dataset": dataset_name,
        "config": config,
        "timings_ms": result["timings"],
        "info": result["info"],
        "shape": list(result["labels"].shape),
    }, indent=2)
