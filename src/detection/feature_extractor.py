"""
Feature extraction from face and hand landmarks.

Combines raw landmark data from both detectors into a
normalised feature dictionary suitable for matching.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np

from ..config import DetectionConfig
from .face_detector import FaceDetector
from .hand_detector import HandDetector

logger = logging.getLogger(__name__)

# ── Landmark index groups ────────────────────────────────────────────
LEFT_EYE_UPPER = [159, 145, 158]
LEFT_EYE_LOWER = [23, 27, 133]
RIGHT_EYE_UPPER = [386, 374, 385]
RIGHT_EYE_LOWER = [253, 257, 362]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [300, 293, 334, 296, 336]
MOUTH_OUTER = [61, 291, 39, 181, 0, 17, 269, 405]
MOUTH_INNER = [78, 308, 95, 88]
NOSE_TIP = 4

FeatureDict = Dict[str, float]


class FeatureExtractor:
    """
    Orchestrates face + hand detectors and produces a feature dict.

    Supports two operating modes:
    - **video** (default): for live webcam frames
    - **image**: for static meme image analysis
    """

    def __init__(self, config: DetectionConfig, *, video_mode: bool = True):
        self._config = config
        self._face = FaceDetector(config, video_mode=video_mode)
        self._hand = HandDetector(config, video_mode=video_mode)
        self._video_mode = video_mode

    # ------------------------------------------------------------------
    def extract(self, bgr_image: np.ndarray) -> Optional[FeatureDict]:
        """
        Return a feature dict for *bgr_image*, or ``None`` if no face
        is detected.
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._video_mode:
            face_res = self._face.detect_for_video(mp_img)
            hand_res = self._hand.detect_for_video(mp_img)
        else:
            face_res = self._face.detect(mp_img)
            hand_res = self._hand.detect(mp_img)

        if not face_res.face_landmarks:
            return None

        landmarks = face_res.face_landmarks[0]
        return self._compute_features(landmarks, hand_res)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pt(landmarks, idx) -> np.ndarray:
        return np.array([landmarks[idx].x, landmarks[idx].y])

    @classmethod
    def _eye_aspect_ratio(cls, landmarks, upper_ids, lower_ids) -> float:
        upper = [cls._pt(landmarks, i) for i in upper_ids]
        lower = [cls._pt(landmarks, i) for i in lower_ids]
        vertical = np.mean(
            [np.linalg.norm(u - l) for u, l in zip(upper, lower)]
        )
        horizontal = np.linalg.norm(
            cls._pt(landmarks, upper_ids[0]) - cls._pt(landmarks, upper_ids[-1])
        )
        return float(vertical / (horizontal + 1e-6))

    def _compute_features(self, landmarks, hand_res) -> FeatureDict:
        pt = lambda idx: self._pt(landmarks, idx)

        # Eye openness
        left_ear = self._eye_aspect_ratio(landmarks, LEFT_EYE_UPPER, LEFT_EYE_LOWER)
        right_ear = self._eye_aspect_ratio(landmarks, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER)
        avg_ear = (left_ear + right_ear) / 2.0

        # Mouth aspect ratio
        mouth_h = float(np.linalg.norm(pt(13) - pt(14)))
        mouth_w = float(np.linalg.norm(pt(61) - pt(291)))
        mouth_ar = mouth_h / (mouth_w + 1e-6)

        # Eyebrow height (distance from brow to eye centre)
        l_brow_y = np.mean([pt(i)[1] for i in LEFT_EYEBROW])
        r_brow_y = np.mean([pt(i)[1] for i in RIGHT_EYEBROW])
        l_eye_cy = np.mean([pt(i)[1] for i in LEFT_EYE_UPPER + LEFT_EYE_LOWER])
        r_eye_cy = np.mean([pt(i)[1] for i in RIGHT_EYE_UPPER + RIGHT_EYE_LOWER])
        avg_brow_height = ((l_eye_cy - l_brow_y) + (r_eye_cy - r_brow_y)) / 2.0

        # Head tilt (roll approximation)
        left_eye_centre = np.mean([pt(i) for i in LEFT_EYE_UPPER + LEFT_EYE_LOWER], axis=0)
        right_eye_centre = np.mean([pt(i) for i in RIGHT_EYE_UPPER + RIGHT_EYE_LOWER], axis=0)
        delta = right_eye_centre - left_eye_centre
        head_tilt = float(np.degrees(np.arctan2(delta[1], delta[0])))

        # Hands
        num_hands = len(hand_res.hand_landmarks) if hand_res.hand_landmarks else 0
        hand_raised = False
        if num_hands > 0:
            face_top = min(l.y for l in landmarks)
            for h_marks in hand_res.hand_landmarks:
                if h_marks[0].y < face_top + 0.3:
                    hand_raised = True
                    break

        return {
            "eye_openness": avg_ear,
            "mouth_openness": mouth_ar,
            "eyebrow_height": float(avg_brow_height),
            "head_tilt": head_tilt,
            "hand_raised": 1.0 if hand_raised else 0.0,
            "num_hands": float(num_hands),
            "surprise_score": avg_ear * float(avg_brow_height) * mouth_ar,
            "smile_score": 1.0 - mouth_ar,
        }

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._face.close()
        self._hand.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
