"""Image filters for the 8-bit windowed slice. Each takes and returns a uint8 gray image."""
import cv2

def _identity(g):
    return g

def _gaussian(g):
    return cv2.GaussianBlur(g, (0, 0), 1.5)

def _median(g):
    return cv2.medianBlur(g, 5)


def _bilateral(g):
    return cv2.bilateralFilter(g, 9, 50, 50)

def _clahe(g):
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)

def _unsharp(g):
    blur = cv2.GaussianBlur(g, (0, 0), 2.0)
    return cv2.addWeighted(g, 1.8, blur, -0.8, 0)

def _sobel(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.convertScaleAbs(mag, alpha=255.0 / max(float(mag.max()), 1e-6))

def _canny(g):
    return cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.0), 50, 150)

# Order here is the order shown in the UI.
FILTERS = {
    "none": _identity,
    "gaussian": _gaussian,     # smoothing / denoising
    "median": _median,         # removes salt-and-pepper style noise, keeps edges
    "bilateral": _bilateral,   # edge-preserving denoising
    "clahe": _clahe,           # local contrast enhancement
    "unsharp": _unsharp,       # sharpening
    "sobel": _sobel,           # edge magnitude
    "canny": _canny,           # edge map
}

def apply_filter(gray, name):
    return FILTERS.get(name, _identity)(gray)
