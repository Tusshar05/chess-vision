"""Export detector + corner model to ONNX (FP32), ONNX INT8 (static QDQ) and CoreML.

    python -m ml.export [--weights-dir weights]

Reads ``weights/manifest.json`` for the active model names and their input sizes, writes
``detector.onnx``, ``detector_int8.onnx``, ``corners.onnx``, ``corners_int8.onnx``,
``*.mlpackage`` (if coremltools works) and ``weights/export_info.json``.
INT8 calibration uses 50 *train/val* images (never the test split).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from ml.chessvision.onnx_infer import letterbox

MODELS = {"detector": "detector", "corners": "corners"}


class _Reader(CalibrationDataReader):
    """Feeds letterboxed calibration images to ONNX Runtime's static quantizer."""

    def __init__(self, input_name: str, images: list[Path], size: int):
        self.it = iter(images)
        self.name, self.size = input_name, size

    def get_next(self) -> dict | None:
        """Next calibration batch or None when exhausted."""
        p = next(self.it, None)
        if p is None:
            return None
        return {self.name: letterbox(cv2.imread(str(p)), self.size)[0]}


def calibration_images(
    root: Path = Path("data/yolo_smoke/detect/images"), n: int = 50
) -> list[Path]:
    """Up to ``n`` train/val images for INT8 calibration."""
    imgs = sorted((root / "train").glob("*.jpg")) + sorted((root / "val").glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError("no calibration images; run `make data` first")
    return imgs[:n]


def export_onnx(pt: Path, imgsz: int, out: Path) -> Path:
    """Export ``pt`` to a fixed-size, simplified ONNX file at ``out``."""
    from ultralytics import YOLO

    exported = Path(
        YOLO(str(pt)).export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False)
    )
    shutil.move(str(exported), out)
    return out


def quantize_int8(fp32: Path, out: Path, imgsz: int, images: list[Path]) -> Path:
    """Static INT8 (QDQ, per-channel) quantization calibrated on ``images``."""
    import onnxruntime as ort

    name = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    quantize_static(
        str(fp32),
        str(out),
        _Reader(name, images, imgsz),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    return out


def export_coreml(pt: Path, imgsz: int, out: Path) -> Path | None:
    """Export to a CoreML ``.mlpackage`` for on-device iPhone inference; None if unsupported."""
    try:
        from ultralytics import YOLO

        p = Path(YOLO(str(pt)).export(format="coreml", imgsz=imgsz, nms=False))
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(p), out)
        return out
    except Exception as e:  # noqa: BLE001
        print("CoreML export skipped:", type(e).__name__, str(e)[:200])
        return None


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-dir", type=Path, default=Path("weights"))
    ap.add_argument("--no-coreml", action="store_true")
    a = ap.parse_args()
    wd = a.weights_dir
    manifest = json.loads((wd / "manifest.json").read_text())
    imgs = calibration_images()
    info: dict = {"kind": manifest["kind"], "models": {}}
    for key in MODELS:
        stem = manifest[key]
        imgsz = manifest.get("imgsz", {}).get(key, 320 if manifest["kind"] == "smoke" else 640)
        pt = wd / f"{stem}.pt"
        fp32 = export_onnx(pt, imgsz, wd / f"{key}.onnx")
        int8 = quantize_int8(fp32, wd / f"{key}_int8.onnx", imgsz, imgs)
        ml = None if a.no_coreml else export_coreml(pt, imgsz, wd / f"{key}.mlpackage")
        info["models"][key] = {
            "source": pt.name,
            "imgsz": imgsz,
            "onnx_fp32_mb": round(fp32.stat().st_size / 1e6, 2),
            "onnx_int8_mb": round(int8.stat().st_size / 1e6, 2),
            "coreml": ml.name if ml else None,
        }
    (wd / "export_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
