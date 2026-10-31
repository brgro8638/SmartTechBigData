"""Data loaders: turn a DICOM series or a video into a point cloud.

Both loaders end in the same format: a DataFrame with three spatial/temporal
axes plus one intensity column, so the clustering and plotting code does not
need to know where the points came from.
"""

import io
import os
import tempfile
import zipfile
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import pydicom


# ---------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------

def _read_datasets_from_zip(zip_bytes):
    """Read every DICOM file inside a zip (any folder depth) that has pixel data."""
    datasets = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for name in archive.namelist():
            if name.endswith("/") or "__MACOSX" in name:
                continue
            try:
                # force=True: many exported series lack the 128-byte DICOM preamble
                ds = pydicom.dcmread(io.BytesIO(archive.read(name)), force=True)
            except Exception:
                continue
            if "PixelData" in ds:
                datasets.append(ds)
    return datasets


def _slice_position(ds):
    """Position along the slice normal, used for sorting and slice spacing.

    Using only the z coordinate works for axial slices but not for sagittal or
    coronal ones, where the slices are stacked along x or y. Projecting the
    position onto the normal (row x column direction) works for any orientation.
    InstanceNumber is the fallback when the geometry tags are missing.
    """
    if "ImagePositionPatient" in ds and "ImageOrientationPatient" in ds:
        orientation = np.array(ds.ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        return float(np.dot(np.array(ds.ImagePositionPatient, dtype=float), normal))
    if "ImagePositionPatient" in ds:
        return float(ds.ImagePositionPatient[2])
    return float(getattr(ds, "InstanceNumber", 0))


def _rescale(ds):
    """Stored values -> real values via the rescale tags.

    For CT this gives Hounsfield units. MR has no physical unit and usually no
    rescale tags, so the values stay as arbitrary signal intensity.
    """
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return ds.pixel_array.astype(np.float32) * slope + intercept


def read_dicom_series(zip_bytes):
    """Read a zip and group its images by series.

    One examination usually holds several series (e.g. a scout, an MRA,
    a FLAIR), and only one of them can be stacked into a single volume.
    Returns a list of dicts sorted by number of images, largest first.
    """
    datasets = _read_datasets_from_zip(zip_bytes)
    if not datasets:
        raise ValueError("No DICOM images with pixel data found in the zip file.")

    groups = {}
    for ds in datasets:
        groups.setdefault(getattr(ds, "SeriesInstanceUID", "unknown"), []).append(ds)

    series = []
    for uid, items in groups.items():
        first = items[0]
        frames = int(getattr(first, "NumberOfFrames", 1))
        series.append({
            "uid": uid,
            "description": str(getattr(first, "SeriesDescription", "")) or "no description",
            "modality": str(getattr(first, "Modality", "")) or "?",
            "images": len(items) if frames == 1 else frames,
            "shape": (int(first.Rows), int(first.Columns)),
            "datasets": items,
        })
    series.sort(key=lambda s: s["images"], reverse=True)
    return series


def build_volume(datasets):
    """Stack one series into a volume.

    Returns (volume, spacing, info): volume is float32 with shape (z, y, x)
    in HU for CT and raw signal for MR; spacing is (dz, dy, dx) in mm.
    """
    modality = str(getattr(datasets[0], "Modality", ""))

    # A single multi-frame file already holds the whole volume
    if len(datasets) == 1 and int(getattr(datasets[0], "NumberOfFrames", 1)) > 1:
        ds = datasets[0]
        volume = _rescale(ds)
        dy, dx = (float(v) for v in getattr(ds, "PixelSpacing", [1.0, 1.0]))
        dz = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
        return volume, (dz, dy, dx), {"slices": volume.shape[0], "modality": modality}

    # Drop slices whose size differs from the majority, otherwise np.stack fails
    shape_counts = Counter((int(ds.Rows), int(ds.Columns)) for ds in datasets)
    main_shape = shape_counts.most_common(1)[0][0]
    datasets = [ds for ds in datasets if (int(ds.Rows), int(ds.Columns)) == main_shape]

    datasets = sorted(datasets, key=_slice_position)
    volume = np.stack([_rescale(ds) for ds in datasets])

    dy, dx = (float(v) for v in getattr(datasets[0], "PixelSpacing", [1.0, 1.0]))
    positions = np.array([_slice_position(ds) for ds in datasets])
    if len(positions) > 1 and np.ptp(positions) > 0:
        # Real slice distance from positions; SliceThickness can differ from spacing
        dz = float(np.median(np.abs(np.diff(positions))))
    else:
        dz = float(getattr(datasets[0], "SliceThickness", 1.0) or 1.0)

    return volume, (dz, dy, dx), {"slices": len(datasets), "modality": modality}


def volume_to_points(volume, spacing, value_min, value_max, stride, max_points, seed,
                     intensity_name="HU"):
    """Keep voxels inside the intensity range and convert them to points in mm.

    stride downsamples the volume first, because argwhere on a full CT
    (hundreds of millions of voxels) would use far too much memory.
    """
    sub = volume[::stride, ::stride, ::stride]
    mask = (sub >= value_min) & (sub <= value_max)
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return pd.DataFrame(columns=["x", "y", "z", intensity_name])

    rng = np.random.default_rng(seed)
    if len(idx) > max_points:
        idx = idx[rng.choice(len(idx), size=max_points, replace=False)]

    dz, dy, dx = spacing
    z, y, x = idx[:, 0], idx[:, 1], idx[:, 2]
    return pd.DataFrame({
        "x": x * stride * dx,
        "y": y * stride * dy,
        "z": z * stride * dz,
        intensity_name: sub[z, y, x],
    })


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def load_video_frames(video_bytes, suffix, resize_width, frame_step, max_frames):
    """Decode a video into a list of small grayscale frames.

    Frames are resized so the number of moving pixels stays manageable.
    Returns (frames, fps).
    """
    # OpenCV can only open videos from a path, not from memory
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        path = tmp.name
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        index = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % frame_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                new_h = max(1, int(round(h * resize_width / w)))
                frames.append(cv2.resize(gray, (resize_width, new_h), interpolation=cv2.INTER_AREA))
            index += 1
        cap.release()
    finally:
        os.remove(path)

    if len(frames) < 2:
        raise ValueError("Could not read at least two frames from the video.")
    return frames, fps


def difference_frames(frames, blur):
    """Absolute difference between consecutive frames (the 'differential frames')."""
    if blur > 1:
        # Blur first so sensor noise does not become thousands of fake moving pixels
        frames = [cv2.GaussianBlur(f, (blur, blur), 0) for f in frames]
    return [cv2.absdiff(frames[i + 1], frames[i]) for i in range(len(frames) - 1)]


def diffs_to_points(diffs, threshold, max_points, seed):
    """Pixels whose change exceeds the threshold become points (x, y, t, diff)."""
    stack = np.stack(diffs)                      # shape (t, y, x)
    idx = np.argwhere(stack > threshold)
    if len(idx) == 0:
        return pd.DataFrame(columns=["x", "y", "t", "diff"])

    rng = np.random.default_rng(seed)
    if len(idx) > max_points:
        idx = idx[rng.choice(len(idx), size=max_points, replace=False)]

    t, y, x = idx[:, 0], idx[:, 1], idx[:, 2]
    return pd.DataFrame({
        "x": x,
        # Flip y so the plot is not upside down (image rows grow downwards)
        "y": stack.shape[1] - 1 - y,
        "t": t,
        "diff": stack[t, y, x].astype(np.float32),
    })
