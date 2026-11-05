"""Reversible pixel prediction applied before entropy coding.

Entropy coders only exploit a skewed symbol distribution. Predicting each pixel
from its neighbours turns spatial redundancy into such a distribution: residuals
cluster around zero. All three coders receive the same stream.
"""

from __future__ import annotations

import numpy as np

PREDICTORS = {
    "None (raw values)": "none",
    "Left neighbour": "left",
    "MED (JPEG-LS)": "med",
}


def forward(image: np.ndarray, predictor: str) -> np.ndarray:
    """Return residuals. Border pixels use whichever neighbour exists."""
    image = image.astype(np.int64)
    if predictor == "none":
        return image
    if image.size == 0:
        return image

    if predictor == "left":
        prediction = np.zeros_like(image)
        prediction[:, 1:] = image[:, :-1]
        prediction[1:, 0] = image[:-1, 0]
        return image - prediction

    if predictor == "med":
        prediction = np.zeros_like(image)
        prediction[0, 1:] = image[0, :-1]
        prediction[1:, 0] = image[:-1, 0]
        left = image[1:, :-1]
        top = image[:-1, 1:]
        corner = image[:-1, :-1]
        # MED picks min/max at an edge, otherwise the planar estimate left + top - corner.
        low = np.minimum(left, top)
        high = np.maximum(left, top)
        prediction[1:, 1:] = np.where(corner >= high, low, np.where(corner <= low, high, left + top - corner))
        return image - prediction

    raise ValueError(f"Unknown predictor: {predictor}")


def inverse(residuals: np.ndarray, predictor: str) -> np.ndarray:
    """Rebuild pixels in raster order: each prediction only uses decoded pixels."""
    residuals = residuals.astype(np.int64)
    if predictor == "none" or residuals.size == 0:
        return residuals

    if predictor == "left":
        image = residuals.copy()
        image[:, 0] = np.cumsum(residuals[:, 0])
        return np.cumsum(image, axis=1)

    if predictor == "med":
        height, width = residuals.shape
        source = residuals.tolist()
        output = [[0] * width for _ in range(height)]
        for row in range(height):
            current = output[row]
            above = output[row - 1] if row else None
            values = source[row]
            for column in range(width):
                if row == 0:
                    predicted = current[column - 1] if column else 0
                elif column == 0:
                    predicted = above[0]
                else:
                    left, top, corner = current[column - 1], above[column], above[column - 1]
                    low, high = (left, top) if left < top else (top, left)
                    if corner >= high:
                        predicted = low
                    elif corner <= low:
                        predicted = high
                    else:
                        predicted = left + top - corner
                current[column] = values[column] + predicted
        return np.array(output, dtype=np.int64)

    raise ValueError(f"Unknown predictor: {predictor}")
