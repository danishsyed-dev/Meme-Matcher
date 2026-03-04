"""
Utility helpers for downloading MediaPipe model files.
"""

from __future__ import annotations

import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_URLS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


def ensure_model(model_filename: str, base_dir: str | Path | None = None) -> str:
    """
    Return the path to *model_filename*, downloading it first if it
    doesn't exist.  Raises ``FileNotFoundError`` on failure.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent  # project root
    else:
        base_dir = Path(base_dir)

    model_path = base_dir / model_filename

    if model_path.exists():
        return str(model_path)

    url = MODEL_URLS.get(model_filename)
    if url is None:
        raise FileNotFoundError(
            f"Unknown model '{model_filename}' and file not found at {model_path}"
        )

    logger.info("Downloading %s from %s …", model_filename, url)
    try:
        subprocess.run(
            ["curl", "-L", url, "-o", str(model_path)],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        # If curl is missing, try urllib as fallback
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(model_path))
        except Exception:
            raise FileNotFoundError(
                f"Failed to download model '{model_filename}': {exc}"
            ) from exc

    if not model_path.exists():
        raise FileNotFoundError(f"Model download failed — {model_path} not found")

    logger.info("Model saved to %s", model_path)
    return str(model_path)
