import os
import cv2
import numpy as np


def compute_fourier_spectrum_ratio(img_bgr):
    """
    Computes a simple high-frequency Fourier spectrum ratio.
    """

    if img_bgr is None or img_bgr.size == 0:
        return 0.70

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    if h < 8 or w < 8:
        return 0.70

    f = np.fft.fftshift(
        np.fft.fft2(gray.astype(np.float32))
    )

    magnitude = 20 * np.log(np.abs(f) + 1e-8)

    cy, cx = h // 2, w // 2
    r = min(h, w) // 4

    y, x = np.ogrid[:h, :w]

    mask = np.ones((h, w), dtype=bool)

    mask[
        (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2
    ] = False

    hf_energy = (
        float(magnitude[mask].mean())
        if mask.any()
        else 0.0
    )

    lf_energy = (
        float(magnitude[~mask].mean())
        if (~mask).any()
        else 1.0
    )

    return hf_energy / (lf_energy + 1e-6)


def check_c2pa_or_synthetic_metadata(filepath):
    """
    Checks media headers for common C2PA or synthetic-media metadata markers.
    """

    if not filepath or not os.path.exists(filepath):
        return False

    try:
        with open(filepath, "rb") as f:
            header = f.read(1024 * 128)

        header_lower = header.lower()

        markers = [
            b"c2pa",
            b"jumdc2pa",
            b"c2pa.assertions",
            b"c2pa.signature",
            b"dall-e",
            b"midjourney",
        ]

        return any(marker in header_lower for marker in markers)

    except Exception:
        return False