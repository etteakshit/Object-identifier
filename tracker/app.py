"""
tracker/app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UniversalTracker — orchestrates all subsystems.

Architecture
────────────
  VideoStream  (daemon thread)  →  frame queue
  ↓
  _preprocess()   CLAHE contrast enhancement
  ↓
  YOLODetector    OpenVINO IR   →  (N, 6) detections
  │               (or PyTorch fallback)
  │  skip-frame logic: reuse last output if inference is slow
  ↓
  ByteTracker     two-stage association  →  (M, 7) tracks
  ↓
  Renderer        boxes + HUD
  ↓
  cv2.imshow()

Full keyboard map
  Q / Esc  quit
  F        focus mode — type class name(s) to highlight
  Esc      clear focus (show all equally)
  M        cycle model  nano → small → medium
  C        cycle confidence threshold
  Z        toggle ROI zoom (centre 60% of frame)
  R        hard-reset all tracks
  S        save screenshot
  H        toggle HUD
  D        toggle DEBUG log level
"""
from __future__ import annotations

import datetime
import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .byte_tracker import ByteTracker
from .config import AppConfig
from .detector import YOLODetector
from .renderer import HUDState, Renderer
from .video_stream import VideoStream

log = logging.getLogger(__name__)


class UniversalTracker:
    """
    Universal Object Detection & Tracking System  (v3).

    Usage
    -----
        cfg = AppConfig()
        app = UniversalTracker(cfg)
        app.run()
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._cfg = config or AppConfig()

        # ── Logging ───────────────────────────────────────────────
        logging.basicConfig(
            level=getattr(logging, self._cfg.log_level, logging.INFO),
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )

        # ── Model / confidence cycling ────────────────────────────
        self._model_idx: int = 1   # default = small (index 1)
        self._conf_idx:  int = 1   # default = 0.30 (index 1)

        # ── Subsystems ────────────────────────────────────────────
        self._stream   = VideoStream(
            camera_index=self._cfg.video.camera_index,
            target_fps=self._cfg.video.target_fps,
            queue_size=self._cfg.video.queue_size,
        )
        self._detector = YOLODetector(self._cfg.detector)
        self._tracker  = ByteTracker(
            max_age=self._cfg.tracker.max_age,
            min_hits=self._cfg.tracker.min_hits,
            iou_threshold=self._cfg.tracker.iou_threshold,
            high_conf_threshold=self._cfg.tracker.high_conf_threshold,
            low_conf_threshold=self._cfg.tracker.low_conf_threshold,
        )
        self._renderer = Renderer()

        # ── Runtime state ─────────────────────────────────────────
        self._focus_classes: set[str]    = set()
        self._roi_zoom:      bool        = False
        self._show_hud:      bool        = True

        # CLAHE processor
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # FPS
        self._fps: float = 0.0
        self._fps_t0:    float = time.time()
        self._fps_count: int   = 0

        # Inference timing
        self._infer_times: deque[float] = deque(maxlen=30)

        # Track count sparkline history
        self._track_history: deque[int] = deque(maxlen=60)
        self._class_counts:  dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=60)
        )

        # Skip-frame state
        self._skip_count: int = 0
        self._last_tracks: np.ndarray = np.empty((0, 7), dtype=float)
        self._last_infer_ms: float    = 0.0

        # Screenshots
        self._screenshot_n: int = 0
        Path(self._cfg.screenshot_dir).mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    # Run
    # ══════════════════════════════════════════════════════════════

    def run(self) -> None:
        """Start detection + tracking loop.  Blocks until quit."""
        self._detector.load()
        self._renderer.warm_colors(self._detector.class_names)

        self._stream.start()
        self._print_welcome()

        cv2.namedWindow(self._cfg.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._cfg.window_title, 1280, 720)

        log.info("Main loop started.")

        try:
            while self._stream.is_alive():
                frame = self._stream.read()
                if frame is None:
                    time.sleep(0.001)   # yield — no new frame yet
                    continue

                frame = self._preprocess(frame)

                # ── Skip-frame decision ───────────────────────────
                should_skip = (
                    self._skip_count < self._cfg.max_skip_frames
                    and len(self._infer_times) >= 5
                    and (
                        sum(list(self._infer_times)[-5:]) / 5
                        > self._cfg.skip_frame_threshold_ms
                    )
                )

                if should_skip:
                    self._skip_count += 1
                    # Let Kalman coast for one frame: feed empty detections
                    self._tracker.update(np.empty((0, 6), dtype=float))
                    tracks   = self._last_tracks   # reuse last display output
                    infer_ms = 0.0
                    log.debug("Frame skipped (skip_count=%d)", self._skip_count)
                else:
                    self._skip_count = 0
                    roi = self._get_roi(frame)
                    dets, infer_ms = self._detector.detect(frame, roi)
                    self._infer_times.append(infer_ms)
                    tracks = self._tracker.update(dets)
                    self._last_tracks   = tracks
                    self._last_infer_ms = infer_ms

                # ── Update history ────────────────────────────────
                total = len(tracks)
                self._track_history.append(total)

                # ── Render ────────────────────────────────────────
                self._renderer.draw_tracks(
                    frame, tracks,
                    self._detector.class_names,
                    self._focus_classes,
                )

                avg_infer = (
                    sum(self._infer_times) / len(self._infer_times)
                    if self._infer_times else 0.0
                )
                hud = HUDState(
                    fps=self._fps,
                    infer_ms=infer_ms,
                    avg_infer_ms=avg_infer,
                    model_label=self._cfg.model_labels[self._model_idx],
                    tracker_name="ByteTrack",
                    backend_name=self._detector.backend_name,
                    track_history=self._track_history,
                    class_counts=self._class_counts,
                    focus_classes=self._focus_classes,
                    roi_zoom=self._roi_zoom,
                    conf_threshold=self._cfg.conf_levels[self._conf_idx],
                    show_hud=self._show_hud,
                    frame_id=self._stream.frame_count,
                    confirmed_count=self._tracker.active_count,
                    lost_count=self._tracker.lost_count,
                    skip_frame=should_skip,
                )
                self._renderer.draw_hud(
                    frame, tracks, self._detector.class_names, hud
                )

                self._tick_fps()
                cv2.imshow(self._cfg.window_title, frame)

                key = cv2.waitKey(1) & 0xFF
                if key != 0xFF and self._handle_key(key, frame):
                    break

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — shutting down.")
        finally:
            self.cleanup()

    # ══════════════════════════════════════════════════════════════
    # Per-frame helpers
    # ══════════════════════════════════════════════════════════════

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        CLAHE contrast enhancement in LAB colour space.

        Locally boosts contrast in under/overexposed regions so YOLO
        receives an image closer to its training distribution.
        Only applied when use_clahe=True (default).
        """
        if not self._cfg.use_clahe:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab_eq = cv2.merge([self._clahe.apply(l), a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _get_roi(
        self, frame: np.ndarray
    ) -> Optional[tuple[int, int, int, int]]:
        """Return the ROI crop tuple if ROI zoom is active, else None."""
        if not self._roi_zoom:
            return None
        h, w = frame.shape[:2]
        xo, yo = int(w * 0.20), int(h * 0.20)
        return (xo, yo, w - xo, h - yo)

    def _tick_fps(self) -> None:
        """Rolling 1-second FPS estimate."""
        self._fps_count += 1
        elapsed = time.time() - self._fps_t0
        if elapsed >= 1.0:
            self._fps       = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_t0    = time.time()

    # ══════════════════════════════════════════════════════════════
    # Keyboard handling
    # ══════════════════════════════════════════════════════════════

    def _handle_key(self, key: int, frame: np.ndarray) -> bool:
        """
        Process a keypress.  Returns True to signal quit.
        """
        ch = chr(key).lower() if key < 128 else ""

        if ch == "q":   # Q or ESC at top-level = quit
            if self._focus_classes and key == 27:
                return self._extracted_from__handle_key_10()
            log.info("Quit requested.")
            return True

        elif key == 27:   # Q or ESC at top-level = quit
            if self._focus_classes:
                return self._extracted_from__handle_key_10()
            log.info("Quit requested.")
            return True

        if ch == "f":
            self._prompt_focus()
        elif ch == "m":
            self._cycle_model()
        elif ch == "c":
            self._conf_idx = (self._conf_idx + 1) % len(self._cfg.conf_levels)
            new_conf = self._cfg.conf_levels[self._conf_idx]
            self._cfg.detector.conf_threshold = new_conf
            log.info("Confidence threshold → %.2f", new_conf)
        elif ch == "z":
            self._roi_zoom = not self._roi_zoom
            log.info("ROI zoom %s", "ON" if self._roi_zoom else "OFF")
        elif ch == "r":
            self._tracker.reset()
            self._track_history.clear()
            self._class_counts.clear()
            self._last_tracks = np.empty((0, 7), dtype=float)
            log.info("Tracker hard-reset.")
        elif ch == "s":
            self._save_screenshot(frame)
        elif ch == "h":
            self._show_hud = not self._show_hud
            log.info("HUD %s", "visible" if self._show_hud else "hidden")
        elif ch == "d":
            new_level = (
                logging.DEBUG
                if logging.getLogger().level != logging.DEBUG
                else logging.INFO
            )
            logging.getLogger().setLevel(new_level)
            log.info("Log level → %s",
                     "DEBUG" if new_level == logging.DEBUG else "INFO")

        return False

    # TODO Rename this here and in `_handle_key`
    def _extracted_from__handle_key_10(self):
        # ESC first clears focus mode; second ESC quits
        self._focus_classes.clear()
        log.info("Focus cleared — tracking all objects.")
        return False

    def _cycle_model(self) -> None:
        """Cycle nano → small → medium and hot-swap the detector."""
        self._model_idx = (self._model_idx + 1) % len(self._cfg.model_paths)
        new_path = self._cfg.model_paths[self._model_idx]
        log.info(
            "Swapping model → %s (%s)",
            new_path, self._cfg.model_labels[self._model_idx],
        )
        self._detector.swap(new_path)
        self._renderer.warm_colors(self._detector.class_names)
        self._tracker.reset()
        self._last_tracks = np.empty((0, 7), dtype=float)

    def _prompt_focus(self) -> None:
        """
        Console prompt to type class name(s) for focus mode.
        Accepts comma-separated names with partial fuzzy matching.
        Runs briefly on the main thread (blocks <1 s).
        """
        names = self._detector.class_names
        known = {v.lower(): k for k, v in names.items()}

        print("\n─── Focus Mode "
              "──────────────────────────────────────────────")
        print("  Enter class name(s) to highlight (comma-separated).")
        print("  Leave blank to clear focus and show all objects equally.")
        preview = ", ".join(sorted(known)[:20])
        print(f"  Known classes (first 20): {preview}  ...")
        print("─────────────────────────────────────────────────────────")
        try:
            raw = input("  >> Focus: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if not raw:
            self._focus_classes.clear()
            log.info("Focus cleared.")
            return

        entries = {c.strip() for c in raw.split(",") if c.strip()}
        valid   = {e for e in entries if e in known}

        # Fuzzy fallback: accept partial matches
        for entry in entries - valid:
            if matches := [k for k in known if entry in k]:
                for m in matches:
                    print(f"  [INFO] '{entry}' → matched '{m}'")
                valid.update(matches)
            else:
                log.warning("Unrecognised class (skipped): '%s'", entry)

        if valid:
            self._focus_classes = valid
            log.info("Focus classes set to: %s", valid)
        else:
            log.warning("No valid classes — focus unchanged.")

    def _save_screenshot(self, frame: np.ndarray) -> None:
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(
            Path(self._cfg.screenshot_dir)
            / f"shot_{ts}_{self._screenshot_n:04d}.png"
        )
        cv2.imwrite(path, frame)
        self._screenshot_n += 1
        log.info("Screenshot saved → %s", path)

    # ══════════════════════════════════════════════════════════════
    # Cleanup
    # ══════════════════════════════════════════════════════════════

    def cleanup(self) -> None:
        """Release all resources. Safe to call multiple times."""
        self._stream.stop()
        cv2.destroyAllWindows()
        log.info("Resources released. Goodbye!")

    # ══════════════════════════════════════════════════════════════
    # Welcome banner
    # ══════════════════════════════════════════════════════════════

    def _print_welcome(self) -> None:
        lines = [
            "═" * 66,
            "   UNIVERSAL OBJECT TRACKER  v3  —  ByteTrack + OpenVINO",
            "═" * 66,
            f"   Detector : {self._cfg.model_labels[self._model_idx]}"
            f"  [{self._detector.backend_name}]",
            f"   Classes  : {len(self._detector.class_names)}",
            f"   Camera   : {self._cfg.video.camera_index}",
            f"   CLAHE    : {'ON' if self._cfg.use_clahe else 'OFF'}",
            f"   Tracker  : ByteTrack"
            f"  max_age={self._cfg.tracker.max_age}"
            f"  high_conf={self._cfg.tracker.high_conf_threshold}",
            "─" * 66,
            "   Q / ESC  — quit",
            "   F        — focus mode  (highlight a class)",
            "   ESC      — clear focus",
            "   M        — swap model  nano → small → medium",
            "   C        — cycle confidence threshold",
            "   Z        — toggle ROI zoom",
            "   R        — reset tracker",
            "   S        — save screenshot",
            "   H        — toggle HUD",
            "   D        — toggle DEBUG logging",
            "═" * 66,
        ]
        for line in lines:
            print(line)
        print()
