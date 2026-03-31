"""
tracker/video_stream.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Threaded frame grabber.

Why threading?
──────────────
cv2.VideoCapture.read() blocks until the camera hardware delivers
the next frame (~16 ms at 60 fps).  If the inference loop calls
read() synchronously, every frame stalls for that 16 ms even when
the GPU / CPU is idle — capping real throughput at 60 / 2 = 30 FPS.

VideoStream runs the grab loop in a background daemon thread and
exposes only the freshest available frame via .read().  The main
loop never blocks; it processes whatever frame the camera most
recently produced.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoStream:
    """
    Non-blocking, threaded camera reader.

    Usage
    -----
        vs = VideoStream(camera_index=0, target_fps=60)
        vs.start()
        while vs.is_alive():
            frame = vs.read()
            if frame is None:
                continue          # no new frame yet — spin or yield
            process(frame)
        vs.stop()
    """

    def __init__(
        self,
        camera_index: int | str = 0,
        target_fps: int = 60,
        queue_size: int = 2,
    ) -> None:
        """
        Args:
            camera_index : cv2.VideoCapture index (int) or RTSP URL (str).
            target_fps   : Requested frame rate hint sent to the driver.
                           Actual rate depends on hardware capability.
            queue_size   : Internal frame ring depth.  Keep at 1–2 so the
                           inference loop always gets the most-recent frame,
                           not one that is several capture cycles stale.
        """
        self.camera_index = camera_index
        self.target_fps   = target_fps
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count: int = 0
        self._started: bool = False

    # ── Public API ──────────────────────────────────────────────────

    def start(self) -> "VideoStream":
        """Open camera and launch capture thread. Returns self for chaining."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera: {self.camera_index!r}.  "
                "Check that the device is connected and not in use."
            )
        # Low-latency driver settings
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))

        actual_fps = self._cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(
            "Camera opened  source=%r  resolution=%dx%d  fps=%.0f",
            self.camera_index, w, h, actual_fps,
        )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="VideoCapture",
        )
        self._thread.start()
        self._started = True
        return self

    def stop(self) -> None:
        """Signal thread to stop and release camera resources."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        log.info("Camera released.")

    def read(self) -> Optional[np.ndarray]:
        """
        Return the most-recent captured frame (non-blocking).

        Drains the internal queue so stale frames never accumulate.
        Returns None if no frame has been captured since the last call.
        """
        latest: Optional[np.ndarray] = None
        with contextlib.suppress(queue.Empty):
            while True:                         # drain all queued frames
                latest = self._q.get_nowait()
        return latest

    def is_alive(self) -> bool:
        """True while the capture thread is running normally."""
        return self._started and not self._stop_event.is_set()

    @property
    def frame_count(self) -> int:
        """Total frames successfully grabbed since start()."""
        return self._frame_count

    # ── Background capture thread ───────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Producer loop: grabs frames as fast as the camera allows and
        pushes them into the bounded queue.  Runs on the daemon thread.
        """
        consecutive_failures = 0
        MAX_FAILURES = 30

        while not self._stop_event.is_set():
            ret, frame = self._cap.read()   # type: ignore[union-attr]

            if not ret:
                consecutive_failures += 1
                log.warning(
                    "Frame grab failed (%d / %d consecutive)",
                    consecutive_failures, MAX_FAILURES,
                )
                if consecutive_failures >= MAX_FAILURES:
                    log.error(
                        "Camera stream failed %d times in a row — stopping.",
                        MAX_FAILURES,
                    )
                    self._stop_event.set()
                    break
                time.sleep(0.02)
                continue

            consecutive_failures = 0
            self._frame_count += 1

            # Evict the oldest entry if the queue is full so the main
            # thread always receives the freshest frame, never a stale one.
            if self._q.full():
                with contextlib.suppress(queue.Empty):
                    self._q.get_nowait()
            with contextlib.suppress(queue.Full):
                self._q.put_nowait(frame)
