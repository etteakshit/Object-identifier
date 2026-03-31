"""
tracker/config.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Typed configuration dataclasses for every subsystem.
All defaults produce a working, high-performance setup.
Edit main.py or pass your own AppConfig to customise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DetectorConfig:
    """
    YOLOv8 inference settings.

    OpenVINO notes
    ──────────────
    Your AMD Ryzen AI 7 has an XDNA2 NPU and AMD Radeon iGPU.
    OpenVINO's GPU plugin targets Intel iGPUs only, so set
    device="CPU" for maximum compatibility.  OpenVINO's CPU
    optimiser (AVX2/AVX-512, threading) still gives ~2-3× the
    throughput of raw PyTorch CPU.

    For AMD GPU acceleration on Windows, swap the Ultralytics
    backend for ONNX Runtime + DirectML (see export_model.py).
    """
    model_path: str = "yolov8s.pt"
    openvino_model_dir: str = ""       # auto-derived if empty
    use_openvino: bool = True          # False → plain PyTorch
    device: str = "CPU"               # "CPU" | "GPU"(Intel) | "NPU"

    # Feed YOLO a low threshold so ByteTrack's second-stage
    # association sees partially-occluded objects at low confidence.
    conf_threshold: float = 0.20

    # Aggressive NMS: boxes that overlap > 20 % are collapsed to
    # one — kills "double detections" of the same physical object.
    nms_iou_threshold: float = 0.20

    def __post_init__(self) -> None:
        if not self.openvino_model_dir:
            stem = Path(self.model_path).stem
            self.openvino_model_dir = f"{stem}_openvino_model"


@dataclass
class TrackerConfig:
    """
    ByteTrack multi-object tracker settings.

    The two-threshold design:
      high_conf_threshold — detections this confident go through
                            stage-1 association (vs. active tracks).
      low_conf_threshold  — weaker detections that pass stage-2
                            association (vs. unmatched tracks).
    Setting low_conf_threshold == DetectorConfig.conf_threshold
    ensures no filtered-by-YOLO box reaches stage 2.
    """
    max_age: int = 40                    # Lost-track coast budget (frames)
    min_hits: int = 2                    # Consecutive hits to confirm a track
    iou_threshold: float = 0.25          # Hungarian gate for all stages
    high_conf_threshold: float = 0.50   # Stage-1 confidence gate
    low_conf_threshold: float = 0.20    # Stage-2 confidence gate


@dataclass
class VideoConfig:
    """Camera / video stream settings."""
    camera_index: int | str = 0   # int index or RTSP URL string
    target_fps: int = 60
    queue_size: int = 2           # Max frames buffered by capture thread


@dataclass
class AppConfig:
    """Top-level application configuration — pass to UniversalTracker."""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    use_clahe: bool = True

    # Skip-frame: if the rolling average inference time exceeds
    # skip_frame_threshold_ms AND we haven't already skipped
    # max_skip_frames consecutive frames, reuse the last track output
    # for one extra frame (Kalman coasts internally).
    skip_frame_threshold_ms: float = 28.0
    max_skip_frames: int = 2

    window_title: str = "Universal Object Tracker — ByteTrack + OpenVINO"
    screenshot_dir: str = "screenshots"
    log_level: str = "INFO"   # "DEBUG" | "INFO" | "WARNING"

    # Confidence cycling  (C key)
    conf_levels: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.65)
    # Model cycling  (M key)
    model_paths: tuple[str, ...] = ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt")
    model_labels: tuple[str, ...] = (
        "Nano (fastest)", "Small (balanced)", "Medium (accurate)"
    )
