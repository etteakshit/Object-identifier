"""
tracker/renderer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All OpenCV drawing logic, extracted into a stateless Renderer
class so app.py focuses entirely on orchestration.

Colour strategy
───────────────
Each YOLO class gets a deterministic BGR colour derived from a
SHA-1 hash of its name — stable across restarts, never random.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_FONTD = cv2.FONT_HERSHEY_DUPLEX


# ─────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────

def _class_color(class_name: str) -> tuple[int, int, int]:
    """Deterministic vivid BGR colour from class name hash."""
    h  = hashlib.sha1(class_name.encode()).digest()
    b  = int(h[0]) % 180 + 60
    g  = int(h[1]) % 180 + 60
    r  = int(h[2]) % 180 + 60
    m  = max(b, g, r)
    if   m == b: b = min(255, b + 80)
    elif m == g: g = min(255, g + 80)
    else:        r = min(255, r + 80)
    return (b, g, r)


def _dim_color(
    color: tuple[int, int, int],
    factor: float = 0.25,
) -> tuple[int, int, int]:
    return tuple(int(c * factor) for c in color)  # type: ignore[return-value]


def _shadow_text(
    frame:  np.ndarray,
    text:   str,
    pos:    tuple[int, int],
    font:   int,
    scale:  float,
    color:  tuple[int, int, int],
    thick:  int = 1,
) -> None:
    """Draw text with a dark drop-shadow for readability on any background."""
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), font, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale,
                color, thick, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
# HUD state (passed per-frame from app.py)
# ─────────────────────────────────────────────────────────────────

@dataclass
class HUDState:
    fps:            float
    infer_ms:       float
    avg_infer_ms:   float
    model_label:    str
    tracker_name:   str
    backend_name:   str
    track_history:  deque
    class_counts:   dict[str, deque]
    focus_classes:  set[str]
    roi_zoom:       bool
    conf_threshold: float
    show_hud:       bool
    frame_id:       int
    confirmed_count: int
    lost_count:      int
    skip_frame:     bool = False


# ─────────────────────────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────────────────────────

class Renderer:
    """
    Stateful renderer holding the colour cache and HUD history.

    Call renderer.render(frame, tracks, hud) each frame.
    """

    def __init__(self) -> None:
        self._color_cache: dict[str, tuple[int, int, int]] = {}

    # ── Colour lookup ─────────────────────────────────────────────

    def warm_colors(self, class_names: dict[int, str]) -> None:
        """Pre-compute class colours after a model load/swap."""
        self._color_cache.clear()
        for name in class_names.values():
            self._color_cache[name.lower()] = _class_color(name.lower())

    def get_color(
        self,
        cls_name:      str,
        focused:       bool,
        focus_active:  bool,
    ) -> tuple[int, int, int]:
        base = self._color_cache.get(cls_name, (180, 180, 180))
        return _dim_color(base, 0.22) if focus_active and not focused else base

    # ── Single box ────────────────────────────────────────────────

    def draw_box(
        self,
        frame:   np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color:   tuple[int, int, int],
        label:   str,
        focused: bool,
        focus_active: bool,
    ) -> None:
        """
        Draw one tracking box with:
          • Glow ring for focused objects
          • Corner tick-marks (tactical HUD style)
          • Thin edge lines between corners
          • Semi-transparent filled label pill
        """
        thick = 3 if focused else 1

        # Glow ring around focused object
        if focused and focus_active:
            glow = tuple(min(255, int(c * 1.4)) for c in color)
            cv2.rectangle(
                frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                glow, 1, cv2.LINE_AA,  # type: ignore[arg-type]
            )

        # Main rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

        # Corner ticks
        t = min(18, int((x2 - x1) * 0.15), int((y2 - y1) * 0.15))
        for (px, py, sx, sy) in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (px, py), (px + sx * t, py),       color, thick + 1)
            cv2.line(frame, (px, py), (px, py + sy * t), color, thick + 1)

        # Label pill
        scale = 0.55 if focused else 0.44
        (lw, lh), bl = cv2.getTextSize(label, _FONT, scale, 1)
        pad = 5
        px1, py1 = x1, max(0, y1 - lh - bl - pad * 2)
        px2, py2 = x1 + lw + pad * 2, y1

        overlay = frame.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), color, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.putText(
            frame, label, (px1 + pad, py2 - bl - 2),
            _FONT, scale, (0, 0, 0), 1, cv2.LINE_AA,
        )

    # ── All tracks in one pass ────────────────────────────────────

    def draw_tracks(
        self,
        frame:         np.ndarray,
        tracks:        np.ndarray,
        class_names:   dict[int, str],
        focus_classes: set[str],
    ) -> None:
        """Draw every confirmed track's box and label onto frame."""
        if not len(tracks):
            return
        focus_active = bool(focus_classes)
        for row in tracks:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            track_id = int(row[4])
            cls_id   = int(row[5])
            conf     = float(row[6])
            cls_name = class_names.get(cls_id, f"cls{cls_id}").lower()
            focused  = not focus_active or cls_name in focus_classes
            color    = self.get_color(cls_name, focused, focus_active)
            label    = f"{cls_name}  ID:{track_id}  {conf:.0%}"
            self.draw_box(frame, x1, y1, x2, y2, color, label, focused, focus_active)

    # ── Full HUD overlay ──────────────────────────────────────────

    def draw_hud(
        self,
        frame:        np.ndarray,
        tracks:       np.ndarray,
        class_names:  dict[int, str],
        hud:          HUDState,
    ) -> None:
        """
        Draw the complete HUD:
          TOP-LEFT    : FPS / inference time / model / backend
          TOP-CENTRE  : Total active track count + skip indicator
          TOP-RIGHT   : Per-class legend panel
          BOTTOM-LEFT : Focus-mode status + sparkline
          BOTTOM-RIGHT: Conf threshold + ROI indicator
        """
        if not hud.show_hud:
            return

        h, w = frame.shape[:2]

        # Count objects per class
        class_count: dict[str, int] = defaultdict(int)
        for row in tracks:
            cls_name = class_names.get(int(row[5]), f"cls{int(row[5])}").lower()
            class_count[cls_name] += 1
        total = sum(class_count.values())

        # ── TOP-LEFT: perf stats ───────────────────────────────────
        fps_color = (
            (0, 255, 180) if hud.fps >= 30 else
            (0, 180, 255) if hud.fps >= 15 else
            (0, 80, 255)
        )
        _shadow_text(frame, f"FPS  {hud.fps:5.1f}", (14, 32),
                     _FONTD, 0.75, fps_color, 2)

        infer_str = (
            "Infer  —  [SKIP]" if hud.skip_frame else f"Infer {hud.avg_infer_ms:.1f} ms"
        )
        _shadow_text(frame, infer_str, (14, 58), _FONT, 0.52, (180, 220, 255))
        _shadow_text(
            frame,
            f"Model: {hud.model_label}  [{hud.backend_name}]",
            (14, 80), _FONT, 0.46, (140, 190, 255),
        )
        _shadow_text(
            frame,
            f"Tracker: {hud.tracker_name}  "
            f"lost={hud.lost_count}",
            (14, 100), _FONT, 0.42, (130, 170, 220),
        )

        # ── TOP-CENTRE: total tracks ───────────────────────────────
        obj_s = "object" if total == 1 else "objects"
        tot_str = f"Tracking  {total}  {obj_s}"
        (tw, _), _ = cv2.getTextSize(tot_str, _FONTD, 0.70, 2)
        _shadow_text(frame, tot_str, ((w - tw) // 2, 34),
                     _FONTD, 0.70, (255, 255, 255), 2)

        # ── TOP-RIGHT: per-class legend ────────────────────────────
        if class_count:
            self._extracted_from_draw_hud_62(class_count, w, frame, hud)
        # ── BOTTOM-LEFT: focus status + sparkline ──────────────────
        if hud.focus_classes:
            _shadow_text(
                frame,
                "Focus: " + ", ".join(sorted(hud.focus_classes)),
                (14, h - 40), _FONT, 0.50, (80, 255, 180),
            )
            _shadow_text(
                frame, "Press ESC to clear focus",
                (14, h - 20), _FONT, 0.42, (130, 200, 150),
            )
        else:
            _shadow_text(
                frame, "All objects  |  F = focus mode",
                (14, h - 16), _FONT, 0.44, (180, 180, 180),
            )

        if len(hud.track_history) > 4:
            self._draw_sparkline(
                frame, hud.track_history, x=14, y=h - 110, width=120, height=36,
            )

        # ── BOTTOM-RIGHT: conf + ROI ───────────────────────────────
        roi_str   = "ROI ON" if hud.roi_zoom else "ROI OFF"
        info_line = f"Conf: {hud.conf_threshold:.2f} [C]   {roi_str} [Z]"
        (iw, _), _ = cv2.getTextSize(info_line, _FONT, 0.46, 1)
        _shadow_text(frame, info_line, (w - iw - 14, h - 16),
                     _FONT, 0.46, (190, 190, 190))

        if hud.roi_zoom:
            xo = int(w * 0.20); yo = int(h * 0.20)
            cv2.rectangle(frame, (xo, yo), (w - xo, h - yo),
                          (80, 255, 80), 1, cv2.LINE_AA)
            cv2.putText(frame, "ROI", (xo + 6, yo + 18),
                        _FONT, 0.50, (80, 255, 80), 1)

    # TODO Rename this here and in `draw_hud`
    def _extracted_from_draw_hud_62(self, class_count, w, frame, hud):
        sorted_cls = sorted(class_count.items(), key=lambda kv: -kv[1])
        PANEL_W = 210
        LINE_H  = 22
        PAD     = 10
        n_lines = min(len(sorted_cls), 12)
        panel_h = n_lines * LINE_H + PAD * 2

        px, py = w - PANEL_W - 10, 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 5, py),
                      (px + PANEL_W, py + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        focus_active = bool(hud.focus_classes)
        for i, (cls_name, cnt) in enumerate(sorted_cls[:12]):
            color   = self._color_cache.get(cls_name, (180, 180, 180))
            focused = not focus_active or cls_name in hud.focus_classes
            if not focused:
                color = _dim_color(color, 0.5)

            row_y = py + PAD + i * LINE_H + 14
            cv2.circle(frame, (px + 10, row_y - 4), 6, color, -1)
            cv2.circle(frame, (px + 10, row_y - 4), 6, (255, 255, 255), 1)

            legend_text = f"{cls_name:<16}  ×{cnt}"
            cv2.putText(
                frame, legend_text, (px + 24, row_y), _FONT, 0.46,
                (255, 255, 255) if focused else (120, 120, 120),
                1, cv2.LINE_AA,
            )

        if len(sorted_cls) > 12:
            cv2.putText(
                frame,
                f"  + {len(sorted_cls) - 12} more...",
                (px + 10, py + panel_h - 4),
                _FONT, 0.40, (150, 150, 150), 1, cv2.LINE_AA,
            )

    # ── Sparkline ─────────────────────────────────────────────────

    def _draw_sparkline(
        self,
        frame:  np.ndarray,
        values: deque,
        x: int, y: int,
        width: int, height: int,
    ) -> None:
        vals  = list(values)
        max_v = max(max(vals), 1)
        ov    = frame.copy()
        cv2.rectangle(ov, (x - 2, y - 2),
                      (x + width + 2, y + height + 2), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        n   = len(vals)
        pts = [
            (x + int(i * width / max(n - 1, 1)),
             y + height - int(v * height / max_v))
            for i, v in enumerate(vals)
        ]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 200, 255), 1)
        cv2.putText(frame, "tracks", (x, y - 5),
                    _FONT, 0.35, (120, 160, 200), 1)
