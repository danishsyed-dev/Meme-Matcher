"""
Configuration loader for Meme Matcher.

Reads config.yaml and provides typed access to all settings
with sensible defaults as fallback.
"""

from __future__ import annotations

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


# ── Validation ────────────────────────────────────────────────────────


class ConfigValidationError(ValueError):
    """Raised when a config value is out of acceptable bounds."""


def validate_config(cfg: AppConfig) -> None:
    """
    Validate all config values are within acceptable bounds.

    Raises ``ConfigValidationError`` with a clear message listing every
    invalid value so the user can fix them all at once.
    """
    errors: List[str] = []

    # Camera
    if not isinstance(cfg.camera.device_index, int) or cfg.camera.device_index < 0:
        errors.append(f"camera.device_index must be a non-negative integer, got {cfg.camera.device_index!r}")
    if not isinstance(cfg.camera.width, int) or cfg.camera.width < 1:
        errors.append(f"camera.width must be a positive integer, got {cfg.camera.width!r}")
    if not isinstance(cfg.camera.height, int) or cfg.camera.height < 1:
        errors.append(f"camera.height must be a positive integer, got {cfg.camera.height!r}")
    if not isinstance(cfg.camera.fps_cap, (int, float)) or cfg.camera.fps_cap < 1 or cfg.camera.fps_cap > 120:
        errors.append(f"camera.fps_cap must be between 1 and 120, got {cfg.camera.fps_cap!r}")

    # Detection — confidence values must be in [0.0, 1.0]
    for name in ("face_confidence", "face_presence_confidence",
                 "hand_confidence", "hand_presence_confidence",
                 "tracking_confidence"):
        val = getattr(cfg.detection, name)
        if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
            errors.append(f"detection.{name} must be a float in [0.0, 1.0], got {val!r}")
    if not isinstance(cfg.detection.max_hands, int) or cfg.detection.max_hands < 0:
        errors.append(f"detection.max_hands must be a non-negative integer, got {cfg.detection.max_hands!r}")

    # Matching
    if not isinstance(cfg.matching.weights, dict):
        errors.append(f"matching.weights must be a dict, got {type(cfg.matching.weights).__name__}")
    else:
        for key, val in cfg.matching.weights.items():
            if not isinstance(val, (int, float)) or val < 0:
                errors.append(f"matching.weights['{key}'] must be a non-negative number, got {val!r}")
    if not isinstance(cfg.matching.debounce_frames, int) or cfg.matching.debounce_frames < 0:
        errors.append(f"matching.debounce_frames must be a non-negative integer, got {cfg.matching.debounce_frames!r}")
    if not isinstance(cfg.matching.decay_factor, (int, float)) or cfg.matching.decay_factor <= 0:
        errors.append(f"matching.decay_factor must be a positive number, got {cfg.matching.decay_factor!r}")

    # UI
    if not isinstance(cfg.ui.window_width, int) or cfg.ui.window_width < 400:
        errors.append(f"ui.window_width must be >= 400, got {cfg.ui.window_width!r}")
    if not isinstance(cfg.ui.window_height, int) or cfg.ui.window_height < 300:
        errors.append(f"ui.window_height must be >= 300, got {cfg.ui.window_height!r}")
    if not isinstance(cfg.ui.display_height, int) or cfg.ui.display_height < 100:
        errors.append(f"ui.display_height must be >= 100, got {cfg.ui.display_height!r}")

    # Assets
    if not isinstance(cfg.assets.folder, str) or not cfg.assets.folder.strip():
        errors.append("assets.folder must be a non-empty string")
    if not isinstance(cfg.assets.supported_formats, list) or not cfg.assets.supported_formats:
        errors.append("assets.supported_formats must be a non-empty list of file extensions")

    if errors:
        bullet_list = "\n  • ".join(errors)
        raise ConfigValidationError(
            f"Invalid config.yaml — {len(errors)} issue(s) found:\n  • {bullet_list}"
        )


# ── Loader ────────────────────────────────────────────────────────────


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    cfg = AppConfig()

    if not config_path.exists():
        validate_config(cfg)
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

    validate_config(cfg)
    return cfg
