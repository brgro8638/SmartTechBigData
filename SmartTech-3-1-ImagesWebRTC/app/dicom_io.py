"""DICOM loading, Hounsfield conversion and window/level.

Also contains a synthetic demo phantom so the app can be tried without any data.
"""
import os
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Series:
    slices: list   # list of float32 2-D arrays (Hounsfield units for CT, raw values otherwise)
    wc: float      # default window centre
    ww: float      # default window width
    source: str    # description of where the data came from

    @property
    def count(self):
        return len(self.slices)

    @property
    def shape(self):
        return self.slices[0].shape

def window(img, wc, ww):
    """Window/level: map [wc - ww/2, wc + ww/2] linearly to 0..255 (uint8)."""
    ww = max(float(ww), 1.0)
    lo = float(wc) - ww / 2.0
    out = (img - lo) * (255.0 / ww)
    return np.clip(out, 0, 255).astype(np.uint8)

def _first(value, default=None):
    """Tags such as WindowCenter may hold one number or a list of numbers."""
    if value is None:
        return default
    try:
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            value = list(value)[0]
        return float(value)
    except (TypeError, ValueError, IndexError):
        return default

def _sort_key(ds):
    pos = getattr(ds, "ImagePositionPatient", None)
    if pos is not None:
        try:
            return float(pos[2])
        except (TypeError, ValueError, IndexError):
            pass
    return float(getattr(ds, "InstanceNumber", 0) or 0)

def load_series(folder):
    """Read every DICOM file under `folder`, keep the largest series, return a Series.
    Returns None if nothing usable was found.
    """
    import pydicom 

    groups = defaultdict(list) # SeriesInstanceUID -> [(sort_key, hu_array, dataset)]
    first_error = None
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            path = os.path.join(root, name)
            try:
                ds = pydicom.dcmread(path, force=True)
                arr = ds.pixel_array
            except Exception as exc:  # not a DICOM, or no pixel data, or unsupported compression
                if first_error is None and name.lower().endswith((".dcm", ".dicom")):
                    first_error = f"{name}: {exc}"
                continue
            if arr.ndim != 2:   # multi-frame / colour images are out of scope for this simple viewer
                continue
            slope = float(getattr(ds, "RescaleSlope", 1) or 1)
            intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
            hu = arr.astype(np.float32) * slope + intercept
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                hu = hu.max() - hu
            groups[str(getattr(ds, "SeriesInstanceUID", ""))].append((_sort_key(ds), hu, ds))

    if not groups:
        if first_error:
            print(f"[dicom] no readable slices. First error: {first_error}")
        return None

    items = max(groups.values(), key=len)
    shape = items[0][1].shape
    items = sorted((it for it in items if it[1].shape == shape), key=lambda it: it[0])

    slices = [it[1] for it in items]
    ds_mid = items[len(items) // 2][2]
    wc = _first(getattr(ds_mid, "WindowCenter", None))
    ww = _first(getattr(ds_mid, "WindowWidth", None))
    if wc is None or ww is None:
        lo, hi = np.percentile(slices[len(slices) // 2], [1, 99])
        wc, ww = (lo + hi) / 2.0, max(hi - lo, 1.0)
    return Series(slices, wc, ww, f"{len(slices)} DICOM slices from {folder}")

def make_demo_series(n=40, size=256):
    """Synthetic CT-like phantom: body, two lungs, spine and a nodule that grows and shrinks."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = size / 2

    def ellipse(cx, cy, rx, ry):
        return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1

    slices = []
    for k in range(n):
        t = k / (n - 1)
        img = np.full((size, size), -1000, np.float32)     # air
        img[ellipse(c, c, 105, 80)] = 40                   # soft tissue
        img[ellipse(c - 45, c - 5, 38, 48)] = -800         # left lung
        img[ellipse(c + 45, c - 5, 38, 48)] = -800         # right lung
        img[ellipse(c, c + 62, 14, 12)] = 700              # spine
        r = 22 * np.sqrt(max(0.0, 1 - ((t - 0.5) / 0.35) ** 2))
        if r > 1:
            img[ellipse(c - 45, c + 5, r, r)] = 90         # nodule inside the left lung
        img = cv2.GaussianBlur(img, (0, 0), 0.8)
        img += rng.normal(0, 12, img.shape).astype(np.float32)
        slices.append(img)
    return Series(slices, 40.0, 400.0, "synthetic demo phantom (no DICOM folder given)")
