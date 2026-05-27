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

# Maximum consecutive read failures before the capture loop gives up.
MAX_READ_RETRIES = 5


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

        # Thread-safe signaling via Event instead of plain bool
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Shared mutable state protected by a lock
        self._lock = threading.Lock()
        self._fps: float = 0.0
        self._error: Optional[str] = None

    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set() and self._thread is not None and self._thread.is_alive()

    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open the camera and begin capturing in a background thread."""
        if self.is_running:
            return

        self._cap = cv2.VideoCapture(self._config.device_index)
        if not self._cap.isOpened():
            with self._lock:
                self._error = (
                    f"Cannot open camera (device {self._config.device_index}). "
                    "Check that a webcam is connected."
                )
            logger.error(self._error)
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)

        self._stop_event.clear()
        with self._lock:
            self._error = None

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera started (device %d)", self._config.device_index)

    def stop(self) -> None:
        """Signal the capture thread to stop and release the camera."""
        self._stop_event.set()
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
        consecutive_failures = 0

        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                logger.warning(
                    "Camera read failed (%d/%d)",
                    consecutive_failures,
                    MAX_READ_RETRIES,
                )
                if consecutive_failures >= MAX_READ_RETRIES:
                    with self._lock:
                        self._error = (
                            f"Camera disconnected after {MAX_READ_RETRIES} "
                            "consecutive read failures."
                        )
                    logger.error(self._error)
                    break
                # Exponential backoff before retrying
                time.sleep(0.1 * consecutive_failures)
                continue

            # Reset failure counter on successful read
            consecutive_failures = 0

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
            with self._lock:
                self._fps = 1.0 / dt if dt > 0 else 0
            prev_time = now

            # Respect FPS cap
            sleep = min_interval - dt
            if sleep > 0:
                time.sleep(sleep)
