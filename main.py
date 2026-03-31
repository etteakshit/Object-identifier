"""
main.py — Entry point for the Universal Object Tracker  (v3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Quick-start
───────────
  # 1. Install dependencies
  pip install -r requirements.txt

  # 2. Export YOLOv8s to OpenVINO IR (run once)
  python export_model.py --model yolov8s.pt

  # 3. Run
  python main.py

Customise any setting by editing the AppConfig below,
or import and subclass AppConfig in your own script.
"""
from tracker import AppConfig, DetectorConfig, TrackerConfig, VideoConfig
from tracker.app import UniversalTracker


def build_config() -> AppConfig:
    """
    Edit this function to tune the tracker for your setup.
    All fields have sensible defaults — only override what you need.
    """
    return AppConfig(

        detector=DetectorConfig(
            # ── Model ─────────────────────────────────────────────
            # yolov8n.pt  → Nano    (fastest; ~25-40+ FPS on OpenVINO CPU)
            # yolov8s.pt  → Small   (balanced; ~15-25 FPS)   ← DEFAULT
            # yolov8m.pt  → Medium  (most accurate; ~8-15 FPS)
            # Press M at runtime to cycle between them live.
            model_path="yolov8s.pt",

            # ── Backend ───────────────────────────────────────────
            # use_openvino=True  requires exporting first (export_model.py).
            # Falls back to PyTorch automatically if the IR dir is missing.
            use_openvino=True,

            # ── Device ────────────────────────────────────────────
            # "CPU"  → Works on any system.  OpenVINO's CPU path
            #          (AVX-512, multi-thread) is still ~2-3× faster
            #          than raw PyTorch CPU — recommended for AMD Ryzen AI.
            # "GPU"  → Intel iGPU only (Arc / Iris Xe).
            #          AMD Radeon is NOT supported by OpenVINO GPU plugin.
            #          For AMD GPU, export to ONNX + use DirectML instead:
            #            python export_model.py --format onnx
            # "NPU"  → Intel NPU (Meteor Lake / Lunar Lake).
            #          AMD XDNA2 is not yet supported by OpenVINO NPU plugin.
            device="CPU",

            # ── Detection thresholds ──────────────────────────────
            # conf_threshold: YOLO minimum confidence to report a box.
            # Set lower (0.20) to feed ByteTrack's low-confidence stage,
            # which recovers occluded objects — the high_conf gate below
            # controls what's actually displayed.
            conf_threshold=0.20,

            # nms_iou_threshold: boxes overlapping > this fraction are
            # collapsed to one via NMS.  0.20 is aggressive — raises the
            # bar for what counts as a "separate" detection, effectively
            # killing the "double-detection" problem for a single object.
            nms_iou_threshold=0.20,
        ),

        tracker=TrackerConfig(
            max_age=40,              # ~1.3 s at 30 fps before track expires
            min_hits=2,              # 2 consecutive frames to confirm a track
            iou_threshold=0.25,      # IoU gate for all Hungarian stages
            high_conf_threshold=0.50,  # Stage-1 gate (shown in bounding box)
            low_conf_threshold=0.20,   # Stage-2 gate (occluded objects)
        ),

        video=VideoConfig(
            camera_index=0,   # 0 = default webcam; also accepts RTSP URLs
            target_fps=60,
            queue_size=2,
        ),

        use_clahe=True,   # CLAHE contrast enhancement — helps in dim lighting

        # ── Skip-frame ────────────────────────────────────────────
        # If the rolling average inference time > threshold, skip YOLO
        # for up to max_skip_frames consecutive frames (Kalman coasts).
        # Prevents inference from blocking the display loop at 60 fps.
        skip_frame_threshold_ms=28.0,
        max_skip_frames=2,

        log_level="INFO",   # "DEBUG" shows per-frame timing; "INFO" is clean
    )


if __name__ == "__main__":
    config = build_config()
    app    = UniversalTracker(config)
    app.run()
