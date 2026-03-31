"""
tracker/detector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OpenVINO-accelerated YOLOv8 detector.

How it works
────────────
Ultralytics natively supports exporting to OpenVINO IR format
(.xml / .bin).  Once exported, loading the model directory
(e.g. "yolov8s_openvino_model/") triggers Ultralytics' OpenVINO
execution provider automatically — no custom pre/post-processing
needed.  All box decoding and NMS happen inside the Ultralytics
wrapper, using the OpenVINO runtime under the hood.

Hardware note for AMD Ryzen AI 7 / Radeon 860M
────────────────────────────────────────────────
OpenVINO's GPU plugin targets Intel iGPUs only.  Set device="CPU"
for AMD setups — OpenVINO's CPU path (AVX2/AVX-512, multi-thread)
still yields ~2-3× more throughput than raw PyTorch CPU inference.

For AMD GPU acceleration on Windows, export to ONNX and use
ONNX Runtime + DirectML (see export_model.py, onnx section).

Run export_model.py once before using use_openvino=True.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class YOLODetector:
    """
    YOLOv8 inference wrapper.

    Transparently switches between:
      • OpenVINO IR model (preferred — run export_model.py first)
      • Plain PyTorch / Ultralytics fallback

    Usage
    -----
        det = YOLODetector(cfg)
        det.load()
        dets, ms = det.detect(frame)
    """

    def __init__(self, config: "DetectorConfig") -> None:  # type: ignore[name-defined]
        from .config import DetectorConfig  # local import avoids circular dep
        self._cfg: DetectorConfig = config
        self._model = None
        self._using_openvino: bool = False

    # ── Model lifecycle ───────────────────────────────────────────

    def load(self) -> None:
        """Load model from disk. Call once at startup."""
        from ultralytics import YOLO  # deferred to avoid slow startup

        ov_dir = Path(self._cfg.openvino_model_dir)

        if self._cfg.use_openvino and ov_dir.exists():
            log.info(
                "Loading OpenVINO IR model from '%s'  device=%s",
                ov_dir, self._cfg.device,
            )
            # Ultralytics detects the directory and engages OpenVINO EP
            self._model = YOLO(str(ov_dir))
            self._using_openvino = True
            log.info("OpenVINO model loaded successfully.")
        else:
            if self._cfg.use_openvino:
                log.warning(
                    "OpenVINO model not found at '%s'. "
                    "Run export_model.py to generate it.  "
                    "Falling back to PyTorch.",
                    ov_dir,
                )
            log.info("Loading PyTorch model '%s'", self._cfg.model_path)
            self._model = YOLO(self._cfg.model_path)
            self._using_openvino = False

        log.info(
            "Detector ready  backend=%s  classes=%d  conf_thresh=%.2f  nms_iou=%.2f",
            self.backend_name,
            len(self.class_names),
            self._cfg.conf_threshold,
            self._cfg.nms_iou_threshold,
        )

    def swap(self, model_path: str) -> None:
        """
        Hot-swap to a different YOLO size (nano / small / medium).
        Automatically picks the OpenVINO export if available.
        """
        self._cfg.model_path = model_path
        stem = Path(model_path).stem
        self._cfg.openvino_model_dir = f"{stem}_openvino_model"
        self.load()
        log.info("Model swapped to '%s'", model_path)

    # ── Inference ─────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        roi: Optional[tuple[int, int, int, int]] = None,
    ) -> tuple[np.ndarray, float]:
        """
        Run YOLOv8 on one frame (or an ROI sub-region).

        Args:
            frame : Full BGR frame from OpenCV.
            roi   : Optional (x_off, y_off, x_end, y_end) pixel crop.
                    Returned detections are in full-frame pixel space.

        Returns:
            detections : (N, 6) float  [x1, y1, x2, y2, conf, cls_id]
            infer_ms   : Wall-clock inference time in milliseconds.
        """
        assert self._model is not None, "Call .load() before .detect()."
        h, w = frame.shape[:2]

        if roi:
            x_off, y_off, x_end, y_end = roi
            input_frame = frame[y_off:y_end, x_off:x_end]
        else:
            x_off = y_off = 0
            input_frame   = frame

        t0 = time.perf_counter()
        results = self._model(
            input_frame,
            verbose=False,
            conf=self._cfg.conf_threshold,
            iou=self._cfg.nms_iou_threshold,   # aggressive NMS
        )[0]
        infer_ms = (time.perf_counter() - t0) * 1e3

        dets: list[list[float]] = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])

            # Translate ROI-relative coords back to full-frame space
            x1 += x_off; x2 += x_off
            y1 += y_off; y2 += y_off

            # Hard-clamp to frame bounds (prevents Kalman state blow-up)
            x1 = max(0., min(x1, w - 1.))
            y1 = max(0., min(y1, h - 1.))
            x2 = max(0., min(x2, w - 1.))
            y2 = max(0., min(y2, h - 1.))

            # Reject degenerate sub-pixel boxes
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue

            dets.append([x1, y1, x2, y2, conf, float(cls_id)])

        det_arr = (
            np.array(dets, dtype=float)
            if dets else np.empty((0, 6), dtype=float)
        )

        log.debug(
            "detect()  infer=%.1f ms  raw_dets=%d  nms_survivors=%d  "
            "backend=%s",
            infer_ms, len(results.boxes), len(dets), self.backend_name,
        )

        return det_arr, infer_ms

    # ── Accessors ─────────────────────────────────────────────────

    @property
    def class_names(self) -> dict[int, str]:
        """YOLO class-index → name mapping."""
        assert self._model is not None, "Call .load() first."
        return self._model.names  # type: ignore[return-value]

    @property
    def backend_name(self) -> str:
        return "OpenVINO" if self._using_openvino else "PyTorch"
