"""Body masking: keep only the patient's body so the scanner table and background cannot be used.

Otsu threshold -> binary opening (breaks thin links to the table) -> largest connected
component -> fill holes (adds the dark lungs back). Pixels outside the mask are set to 0 and
histogram equalisation is then computed inside the mask only.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


def otsu_threshold(pixels: np.ndarray) -> float:
    hist = np.bincount(pixels.ravel(), minlength=256).astype(float)
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return float(np.argmax(between))


def body_mask(image: Image.Image, opening_radius: int = 5) -> np.ndarray:
    pixels = np.asarray(image.convert("L"))
    foreground = pixels > otsu_threshold(pixels)
    structure = np.ones((2 * opening_radius + 1, 2 * opening_radius + 1), dtype=bool)
    foreground = ndimage.binary_opening(foreground, structure=structure)
    labels, count = ndimage.label(foreground)
    if count == 0:
        return np.ones_like(pixels, dtype=bool)
    largest = 1 + int(np.argmax(ndimage.sum(foreground, labels, index=range(1, count + 1))))
    return ndimage.binary_fill_holes(labels == largest)


class BodyMaskEqualize:
    """Zero everything outside the body, then equalise the histogram inside the body only."""

    def __call__(self, image: Image.Image) -> Image.Image:
        gray = image.convert("L")
        mask = body_mask(gray)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        equalised = np.asarray(ImageOps.equalize(gray, mask=mask_image))
        return Image.fromarray(np.where(mask, equalised, 0).astype(np.uint8))
