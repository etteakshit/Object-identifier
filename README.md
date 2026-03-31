# Universal Object Tracker v3
### ByteTrack + OpenVINO + Threaded Capture — Roboyaan Competition

---

## Quick-start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Export YOLOv8s → OpenVINO IR  (run once)
python export_model.py --model yolov8s.pt

# 3. Run
python main.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  VideoStream  (daemon thread)                                    │
│    cv2.VideoCapture.read() → bounded queue (size 2)              │
│    Never blocks the inference loop — always serves freshest      │
│    frame, dropping stale ones.                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │ frame (BGR ndarray)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  _preprocess()   CLAHE in LAB colour space                       │
│  Boosts local contrast → better YOLO accuracy in dim lighting    │
└────────────────────────┬────────────────────────────────────────┘
                         │ enhanced frame
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  YOLODetector                                                    │
│    Backend:  OpenVINO IR  (preferred)  or  PyTorch fallback      │
│    NMS IoU:  0.20  (aggressive — kills double-detections)        │
│    Conf:     0.20  (low feed threshold for ByteTrack stage 2)    │
│                                                                  │
│  Skip-frame logic                                                │
│    If rolling avg inference > 28 ms AND skip budget allows,      │
│    feed tracker empty detections (Kalman coasts) and reuse       │
│    last display output.  Max 2 consecutive skip frames.          │
└────────────────────────┬────────────────────────────────────────┘
                         │ (N, 6) [x1,y1,x2,y2,conf,cls_id]
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  ByteTracker                                                     │
│                                                                  │
│  Split detections:                                               │
│    dets_high  conf ≥ 0.50   (confident detections)               │
│    dets_low   0.20 ≤ conf < 0.50  (uncertain / occluded)         │
│                                                                  │
│  Stage 1: dets_high  ↔  confirmed tracks   (IoU Hungarian)       │
│  Stage 2: dets_low   ↔  unmatched confirmed (IoU Hungarian)      │
│           ^ This is ByteTrack's key innovation: occluded         │
│             objects that fire at low confidence still get         │
│             matched to their existing track.                     │
│  Stage 3: remaining high  ↔  Lost tracks   (re-activation)       │
│  Stage 4: still-unmatched high → new tentative tracks            │
│                                                                  │
│  Track state machine:                                            │
│    Tentative → Confirmed  (after min_hits=2 consecutive)         │
│    Confirmed → Lost       (no detection for ≥ 1 frame)           │
│    Lost      → Confirmed  (re-activated, instantly confirmed)    │
│    Lost      → Dead       (max_age=40 frames exceeded)           │
└────────────────────────┬────────────────────────────────────────┘
                         │ (M, 7) [x1,y1,x2,y2,track_id,cls_id,conf]
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Renderer                                                        │
│    • Per-class deterministic colours (SHA-1 hash, stable)        │
│    • Corner-tick bounding boxes + label pills                    │
│    • Focus mode: dims non-selected classes                       │
│    • HUD: FPS / inference time / model / backend / track legend  │
│    • Sparkline: track count over last 60 frames                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Hardware setup for AMD Ryzen AI 7 + Radeon 860M

| Backend | Speed | Setup |
|---|---|---|
| **OpenVINO CPU** (default) | ~2-3× faster than PyTorch CPU | `python export_model.py` |
| **OpenVINO GPU** | ❌ Intel iGPU only | n/a |
| **ONNX + DirectML** | AMD GPU acceleration (Windows) | See below |
| **PyTorch CPU** | Baseline | No export needed |

### AMD GPU via ONNX Runtime + DirectML

```bash
pip install onnxruntime-directml
python export_model.py --model yolov8s.pt --format onnx
```

Then swap the `YOLODetector` backend in `detector.py` to use an
`onnxruntime.InferenceSession` with `DmlExecutionProvider`.

---

## Keyboard controls

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |
| `F` | Focus mode — type class name(s) to highlight |
| `Esc` | Clear focus (if active) |
| `M` | Cycle model: Nano → Small → Medium |
| `C` | Cycle confidence threshold |
| `Z` | Toggle ROI zoom (centre 60% of frame) |
| `R` | Hard-reset all tracks |
| `S` | Save annotated screenshot |
| `H` | Toggle HUD |
| `D` | Toggle DEBUG logging |

---

## File structure

```
tracker_v3/
├── tracker/
│   ├── __init__.py       — package exports
│   ├── config.py         — AppConfig / DetectorConfig / TrackerConfig
│   ├── video_stream.py   — Threaded VideoStream (daemon capture thread)
│   ├── byte_tracker.py   — ByteTrack + KalmanBoxTracker
│   ├── detector.py       — YOLODetector (OpenVINO / PyTorch)
│   ├── renderer.py       — Renderer + HUDState  (all OpenCV drawing)
│   └── app.py            — UniversalTracker orchestrator
├── main.py               — entry point + AppConfig tuning
├── export_model.py       — one-shot OpenVINO / ONNX export
├── requirements.txt
└── README.md
```

---

## Upgrading from v2 (SORT)

| v2 | v3 |
|---|---|
| `SORTTracker` | `ByteTracker` |
| `sort_tracker.py` | `tracker/byte_tracker.py` |
| `target_tracker.py` | `tracker/app.py` + `tracker/renderer.py` |
| `print()` statements | `logging` (structured, levelled) |
| Blocking `cap.read()` | Threaded `VideoStream` |
| PyTorch inference | OpenVINO IR (with PyTorch fallback) |
| No skip-frame | Skip-frame Kalman coasting |
| NMS IoU = 0.25 | NMS IoU = 0.20 (more aggressive) |
