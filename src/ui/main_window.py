"""
Main window — assembles all UI panels into the root Tkinter window.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from ..config import UIConfig
from .control_panel import ControlPanel
from .status_bar import StatusBar
from .video_panel import VideoPanel


class MainWindow:
    """
    Owns the ``tk.Tk`` root and lays out every panel.

    Call-backs for button actions are injected via the constructor so
    that the window itself stays UI-only — no business logic.
    """

    def __init__(
        self,
        *,
        config: UIConfig,
        on_screenshot: Callable,
        on_upload: Callable,
        on_reload: Callable,
        on_close: Callable,
    ) -> None:
        self.root = tk.Tk()
        self.root.title("Meme Matcher v3.0")
        self.root.geometry(f"{config.window_width}x{config.window_height}")
        self.root.configure(bg="#1E1E1E")
        self.root.protocol("WM_DELETE_WINDOW", on_close)

        self._apply_theme()

        # ── Layout ────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg="#1E1E1E")
        body.pack(fill=tk.BOTH, expand=True, padx=15, pady=(15, 0))

        self.video_panel = VideoPanel(body, display_height=config.display_height)
        self.video_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.control_panel = ControlPanel(
            body,
            on_screenshot=on_screenshot,
            on_upload=on_upload,
            on_reload=on_reload,
        )
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))

        self.status_bar = StatusBar(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # ── Keyboard shortcuts ────────────────────────────────────────
        self.root.bind("<space>", lambda _: on_screenshot())
        self.root.bind("r", lambda _: on_reload())
        self.root.bind("<Escape>", lambda _: on_close())

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_theme() -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background="#1E1E1E")
        style.configure(
            "Dark.TLabel",
            background="#1E1E1E",
            foreground="white",
            font=("Segoe UI", 12),
        )
        style.configure(
            "TProgressbar",
            troughcolor="#333333",
            background="#007ACC",
            thickness=14,
        )

    # ------------------------------------------------------------------
    def mainloop(self) -> None:
        self.root.mainloop()
