"""
One-time script to extract feature vectors from all meme images.

Generates ``web/src/matching/meme-data.json`` for the web app.

Usage::
    python scripts/extract_meme_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path so we can import src.*
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.detection.feature_extractor import FeatureExtractor

import cv2


def main() -> None:
    config = load_config()
    assets_dir = ROOT / config.assets.folder
    formats = set(config.assets.supported_formats)

    print(f"Scanning {assets_dir} for meme images...")

    extractor = FeatureExtractor(config.detection, video_mode=False)
    memes = []

    try:
        for img_path in sorted(assets_dir.iterdir()):
            if img_path.suffix.lower() not in formats:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  SKIP (unreadable): {img_path.name}")
                continue

            features = extractor.extract(img)
            if features is None:
                print(f"  SKIP (no face): {img_path.name}")
                continue

            name = img_path.stem.replace("_", " ").title()
            memes.append({
                "name": name,
                "filename": img_path.name,
                "features": features,
            })
            print(f"  OK: {img_path.name} -> {name}")

    finally:
        extractor.close()

    # Write JSON
    out_path = ROOT / "web" / "src" / "matching" / "meme-data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(memes, f, indent=2)

    print(f"\nDone! Extracted {len(memes)} memes -> {out_path}")


if __name__ == "__main__":
    main()
