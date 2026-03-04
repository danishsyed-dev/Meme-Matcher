"""
Configuration loader for Meme Matcher.

Reads config.yaml and provides typed access to all settings
with sensible defaults as fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps_cap: int = 30


@dataclass
class DetectionConfig:
    face_confidence: float = 0.5
    face_presence_confidence: float = 0.5
    hand_confidence: float = 0.3
    hand_presence_confidence: float = 0.3
    tracking_confidence: float = 0.5
    max_hands: int = 2


@dataclass
class MatchingConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {
        "surprise_score": 20,
        "mouth_openness": 20,
        "hand_raised": 20,
        "eye_openness": 15,
    })
    debounce_frames: int = 5
    decay_factor: float = 5.0


@dataclass
class UIConfig:
    window_width: int = 1400
    window_height: int = 900
    display_height: int = 600
    theme: str = "dark"


@dataclass
class AssetsConfig:
    folder: str = "assets"
    supported_formats: List[str] = field(
        default_factory=lambda: [".jpg", ".png", ".jpeg"]
    )


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    assets: AssetsConfig = field(default_factory=AssetsConfig)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    cfg = AppConfig()

    if not config_path.exists():
        return cfg

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    # Camera
    cam = raw.get("camera", {})
    cfg.camera = CameraConfig(
        device_index=cam.get("device_index", cfg.camera.device_index),
        width=cam.get("width", cfg.camera.width),
        height=cam.get("height", cfg.camera.height),
        fps_cap=cam.get("fps_cap", cfg.camera.fps_cap),
    )

    # Detection
    det = raw.get("detection", {})
    cfg.detection = DetectionConfig(
        face_confidence=det.get("face_confidence", cfg.detection.face_confidence),
        face_presence_confidence=det.get("face_presence_confidence", cfg.detection.face_presence_confidence),
        hand_confidence=det.get("hand_confidence", cfg.detection.hand_confidence),
        hand_presence_confidence=det.get("hand_presence_confidence", cfg.detection.hand_presence_confidence),
        tracking_confidence=det.get("tracking_confidence", cfg.detection.tracking_confidence),
        max_hands=det.get("max_hands", cfg.detection.max_hands),
    )

    # Matching
    mat = raw.get("matching", {})
    cfg.matching = MatchingConfig(
        weights=mat.get("weights", cfg.matching.weights),
        debounce_frames=mat.get("debounce_frames", cfg.matching.debounce_frames),
        decay_factor=mat.get("decay_factor", cfg.matching.decay_factor),
    )

    # UI
    ui = raw.get("ui", {})
    cfg.ui = UIConfig(
        window_width=ui.get("window_width", cfg.ui.window_width),
        window_height=ui.get("window_height", cfg.ui.window_height),
        display_height=ui.get("display_height", cfg.ui.display_height),
        theme=ui.get("theme", cfg.ui.theme),
    )

    # Assets
    ast = raw.get("assets", {})
    cfg.assets = AssetsConfig(
        folder=ast.get("folder", cfg.assets.folder),
        supported_formats=ast.get("supported_formats", cfg.assets.supported_formats),
    )

    return cfg
