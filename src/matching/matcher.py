"""
Meme matching engine.

Loads meme images, pre-computes their feature vectors, and
scores live user features against them with debouncing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import AppConfig
from ..detection.feature_extractor import FeatureDict, FeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class Meme:
    """A loaded meme asset."""

    name: str
    path: str
    image: np.ndarray
    features: FeatureDict


class MatchResult:
    """Holds the best match and its score."""

    __slots__ = ("meme", "score")

    def __init__(self, meme: Optional[Meme], score: float):
        self.meme = meme
        self.score = score


class Matcher:
    """
    Loads memes from the assets folder, pre-analyses them, and
    provides a ``find_match`` method for live feature dicts.

    Supports **match debouncing**: the current match is held for
    ``debounce_frames`` consecutive frames before switching.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._weights = config.matching.weights
        self._decay = config.matching.decay_factor
        self._debounce_limit = config.matching.debounce_frames

        self.memes: List[Meme] = []

        # Debounce state
        self._current_match: Optional[Meme] = None
        self._pending_match: Optional[Meme] = None
        self._pending_count: int = 0

    # ------------------------------------------------------------------
    # Asset loading
    # ------------------------------------------------------------------
    def load_memes(self) -> int:
        """
        (Re)load meme images from the configured assets folder.
        Returns the number of memes loaded.
        """
        self.memes.clear()
        assets_path = Path(self._config.assets.folder)
        if not assets_path.exists():
            assets_path.mkdir(parents=True)
            logger.warning("Created empty assets folder: %s", assets_path)
            return 0

        # Use a *static* (IMAGE-mode) extractor for meme analysis
        extractor = FeatureExtractor(self._config.detection, video_mode=False)

        try:
            files: List[Path] = []
            for ext in self._config.assets.supported_formats:
                files.extend(assets_path.glob(f"*{ext}"))
            files.sort()

            for f in files:
                img = cv2.imread(str(f))
                if img is None:
                    logger.warning("Could not read image: %s", f)
                    continue

                feats = extractor.extract(img)
                if feats is None:
                    logger.info("No face detected in meme: %s — skipping", f.name)
                    continue

                self.memes.append(
                    Meme(
                        name=f.stem.replace("_", " ").title(),
                        path=str(f),
                        image=img,
                        features=feats,
                    )
                )
        finally:
            extractor.close()

        logger.info("Loaded %d memes from %s", len(self.memes), assets_path)
        return len(self.memes)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    def find_match(self, user_feats: Optional[FeatureDict]) -> MatchResult:
        """
        Return the best-matching meme for *user_feats*.

        Applies debouncing so that the returned meme only changes
        after ``debounce_frames`` consecutive frames favouring a
        different meme.
        """
        if user_feats is None or not self.memes:
            return MatchResult(None, 0.0)

        best_meme: Optional[Meme] = None
        best_score = -1.0

        for meme in self.memes:
            score = self._similarity(user_feats, meme.features)
            if score > best_score:
                best_score = score
                best_meme = meme

        # Debounce logic
        if best_meme is not self._current_match:
            if best_meme is self._pending_match:
                self._pending_count += 1
            else:
                self._pending_match = best_meme
                self._pending_count = 1

            if self._pending_count >= self._debounce_limit:
                self._current_match = best_meme
                self._pending_match = None
                self._pending_count = 0
        else:
            self._pending_match = None
            self._pending_count = 0

        return MatchResult(self._current_match, best_score)

    def _similarity(self, a: FeatureDict, b: FeatureDict) -> float:
        score = 0.0
        for key, weight in self._weights.items():
            if key in a and key in b:
                diff = abs(a[key] - b[key])
                score += weight * np.exp(-diff * self._decay)
        return float(score)
