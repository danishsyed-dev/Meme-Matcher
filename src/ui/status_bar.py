"""
Status bar — bottom bar showing FPS, camera state, and meme count.
"""

from __future__ import annotations

import tkinter as tk


class StatusBar(tk.Frame):
    """Thin bar at the bottom of the window."""

    def __init__(self, parent: tk.Widget, **kw):
        super().__init__(parent, bg="#007ACC", height=28, **kw)
        self.pack_propagate(False)

        self._fps_label = tk.Label(
            self, text="FPS: --", bg="#007ACC", fg="white", font=("Consolas", 9)
        )
        self._fps_label.pack(side=tk.LEFT, padx=10)

        self._cam_label = tk.Label(
            self, text="Camera: OFF", bg="#007ACC", fg="white", font=("Consolas", 9)
        )
        self._cam_label.pack(side=tk.LEFT, padx=10)

        self._meme_label = tk.Label(
            self, text="Memes: 0", bg="#007ACC", fg="white", font=("Consolas", 9)
        )
        self._meme_label.pack(side=tk.RIGHT, padx=10)

        self._shortcut_label = tk.Label(
            self,
            text="Space: Screenshot  |  R: Reload  |  Esc: Quit",
            bg="#007ACC",
            fg="#DDDDDD",
            font=("Consolas", 8),
        )
        self._shortcut_label.pack(side=tk.RIGHT, padx=20)

    def set_fps(self, fps: float) -> None:
        self._fps_label.config(text=f"FPS: {fps:.1f}")

    def set_camera_state(self, running: bool) -> None:
        state = "ON" if running else "OFF"
        color = "white" if running else "#FF4444"
        self._cam_label.config(text=f"Camera: {state}", fg=color)

    def set_meme_count(self, count: int) -> None:
        self._meme_label.config(text=f"Memes: {count}")
