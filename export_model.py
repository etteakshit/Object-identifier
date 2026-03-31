"""
export_model.py — One-shot YOLOv8 → OpenVINO IR export
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run ONCE before using use_openvino=True in main.py.

Usage
─────
  # Export all three sizes (recommended)
  python export_model.py --model yolov8n.pt yolov8s.pt yolov8m.pt

  # Export only small
  python export_model.py --model yolov8s.pt

  # Export to ONNX for AMD GPU via ONNX Runtime + DirectML (Windows)
  python export_model.py --model yolov8s.pt --format onnx

Output
──────
  yolov8s_openvino_model/
    ├── yolov8s.xml   (model graph)
    └── yolov8s.bin   (weights)

  The tracker auto-detects these directories at startup.

Hardware guide
──────────────
  CPU  (any)        → use OpenVINO IR  (this script, default)
  Intel iGPU        → use OpenVINO IR  with device="GPU"
  AMD Radeon GPU    → use ONNX + DirectML (--format onnx, see below)
  AMD XDNA2 NPU     → not yet supported by OpenVINO/ONNX Runtime NPU EP;
                       monitor https://github.com/microsoft/onnxruntime/

DirectML quick-start for AMD GPU (Windows)
──────────────────────────────────────────
  pip install onnxruntime-directml
  python export_model.py --model yolov8s.pt --format onnx
  # Then in your custom inference code:
  #   import onnxruntime as ort
  #   sess = ort.InferenceSession("yolov8s.onnx",
  #              providers=["DmlExecutionProvider"])
  # Alternatively: Ultralytics supports predict(device="directml") on some
  # builds — check the latest Ultralytics changelog.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def export_model(model_path: str, fmt: str = "openvino") -> None:
    """
    Export a YOLOv8 .pt checkpoint to the requested format.

    Args:
        model_path : Path to .pt file, e.g. "yolov8s.pt".
                     Ultralytics auto-downloads it if not present.
        fmt        : "openvino" (default) or "onnx".
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed.  Run: pip install ultralytics")
        sys.exit(1)

    model_path = model_path
    log.info("Loading '%s' ...", model_path)
    model = YOLO(model_path)

    if fmt == "onnx":
        log.info("Exporting '%s' → ONNX format ...", model_path)
        export_path = model.export(
            format="onnx",
            dynamic=False,
            simplify=True,
            opset=17,
        )
        log.info("Export complete → '%s'", export_path)
        log.info(
            "Use with ONNX Runtime + DirectML for AMD GPU acceleration.\n"
            "  pip install onnxruntime-directml\n"
            "  providers=['DmlExecutionProvider', 'CPUExecutionProvider']"
        )
    elif fmt == "openvino":
        _extracted_from_export_model_21(model_path, model)
    else:
        log.error("Unknown format '%s'. Choose 'openvino' or 'onnx'.", fmt)
        sys.exit(1)


# TODO Rename this here and in `export_model`
def _extracted_from_export_model_21(model_path, model):
    log.info(
        "Exporting '%s' → OpenVINO IR format ...", model_path
    )
    # dynamic=False gives a fixed-shape model which is faster on CPU.
    # half=False keeps FP32; set half=True if your CPU supports FP16 AVX.
    export_path = model.export(
        format="openvino",
        dynamic=False,
        half=False,
        simplify=True,
    )
    stem    = Path(model_path).stem
    out_dir = f"{stem}_openvino_model"
    log.info(
        "Export complete → '%s'", out_dir
    )
    log.info(
        "Set use_openvino=True and device='CPU' in main.py to use it."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 to OpenVINO IR or ONNX."
    )
    parser.add_argument(
        "--model", nargs="+",
        default=["yolov8s.pt"],
        help="One or more .pt checkpoint paths (default: yolov8s.pt)",
    )
    parser.add_argument(
        "--format", choices=["openvino", "onnx"],
        default="openvino",
        help="Export format (default: openvino)",
    )
    args = parser.parse_args()

    for mp in args.model:
        export_model(mp, fmt=args.format)


if __name__ == "__main__":
    main()
