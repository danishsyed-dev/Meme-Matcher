"""
Application controller — wires together camera, detection, matching, and UI.

This is the single orchestration point: it owns the main loop
timer (``root.after``) that polls the camera queue, runs detection,
and pushes results to the UI.
"""

from __future__ import annotations

import datetime
import logging
import shutil
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import cv2
import numpy as np

from .camera.camera_manager import CameraManager
from .config import AppConfig, load_config
from .detection.feature_extractor import FeatureExtractor
from .matching.matcher import Matcher
from .ui.main_window import MainWindow

logger = logging.getLogger(__name__)

POLL_INTERVAL_MS = 15  # ~66 FPS polling rate


class App:
    """Top-level application controller."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._config = config or load_config()

        # Backend components
        self._camera = CameraManager(self._config.camera)
        self._extractor = FeatureExtractor(self._config.detection, video_mode=True)
        self._matcher = Matcher(self._config)

        # Last rendered combined frame (for screenshots)
        self._last_frame: Optional[np.ndarray] = None

        # UI
        self._window = MainWindow(
            config=self._config.ui,
            on_screenshot=self._save_screenshot,
            on_upload=self._upload_meme,
            on_reload=self._reload_assets,
            on_close=self._shutdown,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Initialise everything and enter the Tkinter main loop."""
        # Load memes
        count = self._matcher.load_memes()
        self._window.control_panel.update_gallery(
            [m.name for m in self._matcher.memes]
        )
        self._window.status_bar.set_meme_count(count)

        # Start camera
        self._camera.start()
        if self._camera.error:
            self._window.video_panel.show_error(self._camera.error)
        self._window.status_bar.set_camera_state(self._camera.is_running)

        # Begin the frame-processing timer
        self._schedule_tick()

        # Enter event loop (blocks until window closes)
        self._window.mainloop()

    # ------------------------------------------------------------------
    # Frame processing (runs on the main thread via root.after)
    # ------------------------------------------------------------------
    def _schedule_tick(self) -> None:
        self._window.root.after(POLL_INTERVAL_MS, self._tick)

    def _tick(self) -> None:
        frame = self._camera.get_frame()

        if frame is not None:
            # Detection
            user_feats = self._extractor.extract(frame)
            result = self._matcher.find_match(user_feats)

            meme_image = result.meme.image if result.meme else None
            self._window.video_panel.update_frame(frame, meme_image)
            self._last_frame = frame  # store for screenshot (raw camera only)

            if result.meme:
                self._window.control_panel.set_match(result.meme.name, result.score)
            else:
                self._window.control_panel.clear_match()

        # Update status bar continuously
        self._window.status_bar.set_fps(self._camera.fps)
        self._window.status_bar.set_camera_state(self._camera.is_running)

        # Reschedule
        self._schedule_tick()

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------
    def _save_screenshot(self) -> None:
        if self._last_frame is None:
            messagebox.showwarning("Screenshot", "No frame available yet.")
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"screenshot_{ts}.jpg"
        cv2.imwrite(path, self._last_frame)
        messagebox.showinfo("Saved", f"Screenshot saved to {path}")

    def _upload_meme(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )
        if not path:
            return

        dest = Path(self._config.assets.folder) / Path(path).name
        shutil.copy(path, dest)
        self._reload_assets()

    def _reload_assets(self) -> None:
        count = self._matcher.load_memes()
        self._window.control_panel.update_gallery(
            [m.name for m in self._matcher.memes]
        )
        self._window.status_bar.set_meme_count(count)
        messagebox.showinfo("Reloaded", f"Loaded {count} memes!")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def _shutdown(self) -> None:
        logger.info("Shutting down…")
        self._camera.stop()
        self._extractor.close()
        self._window.root.destroy()
