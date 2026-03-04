"""
Video panel — displays the live camera feed and matched meme.
"""

from __future__ import annotations

import tkinter as tk
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

from ..utils.image_utils import combine_side_by_side, cv2_to_pil, resize_keep_aspect


class VideoPanel(tk.Frame):
    """
    Left-side panel that shows the combined camera + meme view.
    """

    def __init__(self, parent: tk.Widget, display_height: int = 600, **kw):
        super().__init__(parent, bg="black", highlightbackground="#333", highlightthickness=2, **kw)
        self._display_height = display_height

        self._label = tk.Label(self, bg="black")
        self._label.pack(fill=tk.BOTH, expand=True)

        # Keep a reference so GC doesn't collect the image
        self._photo: Optional[ImageTk.PhotoImage] = None

        # Placeholder text
        self._label.config(
            text="Waiting for camera…",
            fg="#555",
            font=("Segoe UI", 16),
        )

    def update_frame(
        self,
        camera_frame: np.ndarray,
        meme_image: Optional[np.ndarray] = None,
    ) -> None:
        """Render a new frame (optionally combined with a meme)."""
        if meme_image is not None:
            combined = combine_side_by_side(camera_frame, meme_image)
        else:
            combined = camera_frame

        pil_img = cv2_to_pil(combined)
        pil_img = resize_keep_aspect(pil_img, self._display_height)

        self._photo = ImageTk.PhotoImage(image=pil_img)
        self._label.config(image=self._photo, text="")

    def show_error(self, message: str) -> None:
        self._label.config(
            image="",
            text=message,
            fg="#FF4444",
            font=("Segoe UI", 14, "bold"),
        )
