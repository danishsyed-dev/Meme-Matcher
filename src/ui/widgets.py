"""
Reusable custom Tkinter widgets.
"""

from __future__ import annotations

import tkinter as tk


class ModernButton(tk.Canvas):
    """
    A flat, hover-aware button drawn on a Canvas.

    Supports keyboard navigation: focusable via Tab, activatable via
    Return or Space, with a visible focus ring.
    """

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
        focus_color: str = "#007ACC",
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
            takefocus=True,  # Keyboard focusable
        )
        self._command = command
        self._bg = bg_color
        self._hover = hover_color
        self._focus_color = focus_color
        self._radius = corner_radius
        self._width = width
        self._height = height

        # Draw rounded rectangle background
        self._rect = self._rounded_rect(2, 2, width - 2, height - 2, corner_radius, fill=bg_color, outline="")
        self._text = self.create_text(
            width / 2, height / 2, text=text, fill=text_color, font=font
        )

        # Focus ring (hidden by default)
        self._focus_ring = self._rounded_rect(
            0, 0, width, height, corner_radius + 1,
            fill="", outline=focus_color, width=2,
        )
        self.itemconfigure(self._focus_ring, state="hidden")

        # Mouse events
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

        # Keyboard events — Return and Space activate the button
        self.bind("<Return>", self._on_click)
        self.bind("<space>", self._on_click)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)

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
            try:
                self._command()
            except Exception:
                # Prevent callback errors from crashing the Tk event loop
                import logging
                logging.getLogger(__name__).exception("Button callback error")

    def _on_focus_in(self, _event):
        self.itemconfigure(self._focus_ring, state="normal")

    def _on_focus_out(self, _event):
        self.itemconfigure(self._focus_ring, state="hidden")
