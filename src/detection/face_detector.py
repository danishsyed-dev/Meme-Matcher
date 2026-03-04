"""
Face landmark detection using MediaPipe Face Landmarker.

Provides a thin wrapper that manages model lifecycle and
exposes a ``detect`` / ``detect_for_video`` interface.
"""

from __future__ import annotations

import logging
from typing import Optional

import mediapipe as mp
import numpy as np

from ..config import DetectionConfig
from ..utils.model_downloader import ensure_model

logger = logging.getLogger(__name__)

_VisionRunningMode = mp.tasks.vision.RunningMode


class FaceDetector:
    """Wraps MediaPipe FaceLandmarker for either IMAGE or VIDEO mode."""

    def __init__(
        self,
        config: DetectionConfig,
        *,
        video_mode: bool = True,
    ) -> None:
        self._config = config
        self._video_mode = video_mode
        self._frame_counter = 0

        model_path = ensure_model("face_landmarker.task")
        running_mode = (
            _VisionRunningMode.VIDEO if video_mode else _VisionRunningMode.IMAGE
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=config.face_confidence,
            min_face_presence_confidence=config.face_presence_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )
        self._detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    def detect(self, mp_image: mp.Image):
        """Run detection on a single image (IMAGE mode)."""
        return self._detector.detect(mp_image)

    def detect_for_video(self, mp_image: mp.Image) -> object:
        """Run detection for one video frame (VIDEO mode)."""
        self._frame_counter += 1
        return self._detector.detect_for_video(mp_image, self._frame_counter)

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
