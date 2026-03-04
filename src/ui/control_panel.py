"""
Control panel — sidebar with match info, buttons, stats graph, and meme gallery.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, List

from .widgets import ModernButton


class ControlPanel(tk.Frame):
    """Right-side panel with controls, match info, and meme gallery."""

    def __init__(
        self,
        parent: tk.Widget,
        *,
        on_screenshot: Callable,
        on_upload: Callable,
        on_reload: Callable,
        width: int = 400,
        **kw,
    ):
        super().__init__(parent, bg="#252526", width=width, **kw)
        self.pack_propagate(False)

        self._build_header()
        self._build_match_section()
        self._build_stats_section()
        self._build_buttons(on_screenshot, on_upload, on_reload)
        self._build_gallery()

        self._history: List[float] = []

    # ── Sections ──────────────────────────────────────────────────────

    def _build_header(self) -> None:
        tk.Label(
            self,
            text="CONTROL CENTER",
            bg="#252526",
            fg="#007ACC",
            font=("Consolas", 16, "bold"),
        ).pack(pady=(20, 10))

    def _build_match_section(self) -> None:
        self._match_label = tk.Label(
            self,
            text="No Match",
            bg="#252526",
            fg="white",
            font=("Segoe UI", 14, "bold"),
        )
        self._match_label.pack(pady=(10, 5))

        self._score_bar = ttk.Progressbar(
            self, orient="horizontal", length=300, mode="determinate"
        )
        self._score_bar.pack(pady=5)

    def _build_stats_section(self) -> None:
        tk.Label(
            self,
            text="Expression History",
            bg="#252526",
            fg="#888888",
            font=("Segoe UI", 9),
        ).pack(pady=(20, 5))

        self._stats_canvas = tk.Canvas(
            self, width=350, height=100, bg="#333333", highlightthickness=0
        )
        self._stats_canvas.pack(pady=5)

    def _build_buttons(self, on_screenshot, on_upload, on_reload) -> None:
        frame = tk.Frame(self, bg="#252526")
        frame.pack(pady=20, fill=tk.X, padx=20)

        for text, cmd in [
            ("\U0001F4F7  Save Screenshot", on_screenshot),
            ("\U0001F4C2  Upload Custom Meme", on_upload),
            ("\U0001F504  Reload Assets", on_reload),
        ]:
            ModernButton(frame, text, cmd, width=340).pack(pady=5, fill=tk.X)

    def _build_gallery(self) -> None:
        tk.Label(
            self,
            text="Active Memes",
            bg="#252526",
            fg="#888888",
            font=("Segoe UI", 9),
        ).pack(pady=(20, 10))

        self._gallery_frame = tk.Frame(self, bg="#252526")
        self._gallery_frame.pack(fill=tk.BOTH, expand=True)

    # ── Public API ────────────────────────────────────────────────────

    def set_match(self, name: str, score: float) -> None:
        self._match_label.config(text=f"{name}  ({int(score)})")
        self._score_bar["value"] = min(score, 100)

        self._history.append(score)
        if len(self._history) > 50:
            self._history.pop(0)
        self._draw_stats()

    def clear_match(self) -> None:
        self._match_label.config(text="No Match")
        self._score_bar["value"] = 0

    def update_gallery(self, meme_names: List[str]) -> None:
        """Rebuild the scrollable gallery with current meme names."""
        for child in self._gallery_frame.winfo_children():
            child.destroy()

        canvas = tk.Canvas(self._gallery_frame, bg="#252526", highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            self._gallery_frame, orient="vertical", command=canvas.yview
        )
        inner = tk.Frame(canvas, bg="#252526")

        inner.bind(
            "<Configure>",
            lambda _: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for name in meme_names:
            row = tk.Frame(inner, bg="#333333", pady=5, padx=8)
            row.pack(fill=tk.X, padx=5, pady=2)
            tk.Label(
                row, text=name, bg="#333333", fg="white", font=("Segoe UI", 9)
            ).pack(anchor="w")

    # ── Stats Graph ───────────────────────────────────────────────────

    def _draw_stats(self) -> None:
        self._stats_canvas.delete("all")
        if not self._history:
            return

        w, h = 350, 100
        step = w / 50

        points: List[float] = []
        for i, val in enumerate(self._history):
            x = i * step
            y = h - (val / 100 * h)
            points.extend([x, y])

        if len(points) >= 4:
            self._stats_canvas.create_line(
                points, fill="#007ACC", width=2, smooth=True
            )
