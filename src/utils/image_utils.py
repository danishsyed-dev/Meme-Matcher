"""
Image utility helpers — resize, color conversion, aspect-ratio fitting.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert a BGR OpenCV image to an RGB PIL Image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_keep_aspect(image: Image.Image, target_height: int) -> Image.Image:
    """Resize a PIL image to *target_height* while preserving aspect ratio."""
    aspect = image.width / image.height
    new_width = int(target_height * aspect)
    return image.resize((new_width, target_height), Image.Resampling.LANCZOS)


def resize_cv2_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """Resize an OpenCV image to *target_height* keeping aspect ratio."""
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height))


def combine_side_by_side(
    left: np.ndarray, right: np.ndarray, divider_color=(0, 255, 255)
) -> np.ndarray:
    """
    Place two images side by side.  *right* is scaled to match the
    height of *left*.  A thin coloured divider is drawn between them.
    """
    h, w_left = left.shape[:2]
    right_resized = resize_cv2_to_height(right, h)
    w_right = right_resized.shape[1]

    combined = np.zeros((h, w_left + w_right, 3), dtype=np.uint8)
    combined[:, :w_left] = left
    combined[:, w_left:] = right_resized

    cv2.line(combined, (w_left, 0), (w_left, h), divider_color, 2)
    return combined
