"""Segmentation: region growing and marker-controlled watershed.

All functions take an 8-bit gray image and return a uint8 mask (255 = segmented region).
"""
import cv2
import numpy as np

def _in_bounds(gray, seed):
    h, w = gray.shape
    return seed is not None and 0 <= seed[0] < w and 0 <= seed[1] < h

def region_growing(gray, seed, tol=10):
    """Grow a region from `seed`, accepting 4-connected pixels within +/- tol of the seed value."""
    if not _in_bounds(gray, seed):
        return np.zeros_like(gray)
    h, w = gray.shape
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)   # so one noisy seed pixel doesn't decide everything
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(smooth, mask, (int(seed[0]), int(seed[1])), 0, (tol,) * 3, (tol,) * 3, flags)
    return mask[1:-1, 1:-1].copy()

def watershed(gray, seed=None, tol=10):
    """Watershed segmentation.

    With a seed: marker-controlled watershed. The seed is the foreground marker, pixels whose
    intensity is far from the seed's are the background marker, and the watershed finds the
    boundary in between.
    Without a seed: automatic (Otsu threshold + distance transform markers).
    """
    bgr = cv2.cvtColor(cv2.GaussianBlur(gray, (0, 0), 1.0), cv2.COLOR_GRAY2BGR)
    if seed is None:
        return _watershed_auto(gray, bgr)
    if not _in_bounds(gray, seed):
        return np.zeros_like(gray)

    h, w = gray.shape
    x, y = int(seed[0]), int(seed[1])
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.5).astype(np.float32)
    ref = float(smooth[y, x])

    markers = np.zeros((h, w), np.int32)
    markers[np.abs(smooth - ref) > max(3 * tol, 30)] = 1   # sure background
    markers[0, :] = markers[-1, :] = markers[:, 0] = markers[:, -1] = 1
    cv2.circle(markers, (x, y), 3, 2, -1)                    # sure foreground (the seed)

    cv2.watershed(bgr, markers)
    return ((markers == 2).astype(np.uint8)) * 255

def _watershed_auto(gray, bgr):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * float(dist.max()), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1            # background becomes 1, regions 2..n
    markers[unknown == 255] = 0      # undecided zone: the watershed fills it in
    cv2.watershed(bgr, markers)
    return ((markers > 1).astype(np.uint8)) * 255

def overlay(gray, mask, seed=None):
    """Return a BGR image: the slice with a translucent red fill, a yellow outline and the seed."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mask is not None and mask.any():
        tint = bgr.copy()
        tint[mask > 0] = (0, 0, 255)
        bgr = cv2.addWeighted(tint, 0.35, bgr, 0.65, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, (0, 255, 255), 1)
    if seed is not None:
        cv2.drawMarker(bgr, (int(seed[0]), int(seed[1])), (0, 255, 0), cv2.MARKER_CROSS, 11, 1)
    return bgr
