"""
Reusable custom Tkinter widgets.
"""

from __future__ import annotations

import tkinter as tk


class ModernButton(tk.Canvas):
    """A flat, hover-aware button drawn on a Canvas."""

    def __init__(
        self,
        parent: tk.Widget,
        text: str,
        command,
        *,
        width: int = 200,
        height: int = 40,
        bg_color: str = "#333333",
        hover_color: str = "#444444",
        text_color: str = "white",
        font: tuple = ("Segoe UI", 10, "bold"),
        corner_radius: int = 6,
    ):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=parent.cget("bg"),
            highlightthickness=0,
        )
        self._command = command
        self._bg = bg_color
        self._hover = hover_color
        self._radius = corner_radius

        # Draw rounded rectangle background
        self._rect = self._rounded_rect(2, 2, width - 2, height - 2, corner_radius, fill=bg_color, outline="")
        self._text = self.create_text(
            width / 2, height / 2, text=text, fill=text_color, font=font
        )

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _rounded_rect(self, x1, y1, x2, y2, r, **kw):
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kw)

    def _on_enter(self, _event):
        self.itemconfigure(self._rect, fill=self._hover)

    def _on_leave(self, _event):
        self.itemconfigure(self._rect, fill=self._bg)

    def _on_click(self, _event):
        if self._command:
            self._command()
