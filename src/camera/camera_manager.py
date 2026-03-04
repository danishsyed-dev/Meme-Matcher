"""
Thread-safe camera manager.

Captures frames in a background thread and exposes them via a
``queue.Queue`` so the main (UI) thread can consume them safely.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from ..config import CameraConfig

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Opens a webcam and continuously pushes frames into a queue.

    Usage::

        cam = CameraManager(config.camera)
        cam.start()
        ...
        frame = cam.get_frame()   # non-blocking; returns None if empty
        ...
        cam.stop()
    """

    def __init__(self, config: CameraConfig, max_queue_size: int = 2) -> None:
        self._config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fps: float = 0.0
        self._error: Optional[str] = None

    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def error(self) -> Optional[str]:
        return self._error

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open the camera and begin capturing in a background thread."""
        if self._running:
            return

        self._cap = cv2.VideoCapture(self._config.device_index)
        if not self._cap.isOpened():
            self._error = (
                f"Cannot open camera (device {self._config.device_index}). "
                "Check that a webcam is connected."
            )
            logger.error(self._error)
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)

        self._running = True
        self._error = None
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera started (device %d)", self._config.device_index)

    def stop(self) -> None:
        """Signal the capture thread to stop and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera stopped")

    # ------------------------------------------------------------------
    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame (non-blocking), or ``None``."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        min_interval = 1.0 / self._config.fps_cap if self._config.fps_cap > 0 else 0
        prev_time = time.perf_counter()

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                self._error = "Camera read failed"
                logger.warning(self._error)
                break

            frame = cv2.flip(frame, 1)  # Mirror for user-facing display

            # Drop stale frames — always keep only the latest
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)

            # FPS calculation
            now = time.perf_counter()
            dt = now - prev_time
            self._fps = 1.0 / dt if dt > 0 else 0
            prev_time = now

            # Respect FPS cap
            sleep = min_interval - dt
            if sleep > 0:
                time.sleep(sleep)

        self._running = False
