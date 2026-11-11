"""PyTorch (Ultralytics) backends with the same interface as the ONNX ones.

Used for evaluation and as the PyTorch row of the benchmark. NOT imported by the API image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.chessvision.board import Detection, order_corners
from ml.chessvision.onnx_infer import CornerResult


class TorchDetector:
    """Ultralytics ``.pt`` detector."""

    def __init__(
        self,
        path: str | Path,
        imgsz: int,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(path), task="detect")
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        """Detect pieces in a BGR image."""
        r = self.model.predict(
            img_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy, cls, conf = b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy()
        return [
            Detection(*map(float, xy), int(c), float(p))
            for xy, c, p in zip(xyxy, cls, conf, strict=True)
        ]


class TorchCornerModel:
    """Ultralytics ``.pt`` board-corner pose model."""

    def __init__(self, path: str | Path, imgsz: int, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(str(path), task="pose")
        self.imgsz, self.device = imgsz, device

    def find(self, img_bgr: np.ndarray, min_conf: float = 0.05) -> CornerResult | None:
        """Return the most confident board's corners, or None."""
        r = self.model.predict(
            img_bgr, imgsz=self.imgsz, conf=min_conf, device=self.device, verbose=False
        )[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return None
        i = int(r.boxes.conf.argmax())
        return CornerResult(order_corners(r.keypoints.xy[i].cpu().numpy()), float(r.boxes.conf[i]))
