"""ONNX Runtime inference for the piece detector and the board-corner model (no PyTorch needed)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ml.chessvision.board import Detection, order_corners


@dataclass
class CornerResult:
    """Board corners in image pixels, clockwise from the image top-left, plus model confidence."""

    corners: np.ndarray  # (4, 2)
    conf: float


def letterbox(img_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize with aspect ratio kept and pad to ``size``x``size`` (grey 114, like Ultralytics).

    Returns ``(nchw float32 RGB in [0,1], scale, (pad_x, pad_y))``.
    """
    h, w = img_bgr.shape[:2]
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    resized = cv2.resize(
        img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py : py + nh, px : px + nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), s, (float(px), float(py))


def make_session(path: str | Path, threads: int | None = None) -> ort.InferenceSession:
    """CPU ONNX Runtime session."""
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _input_size(sess: ort.InferenceSession) -> int:
    return int(sess.get_inputs()[0].shape[-1])


class OnnxDetector:
    """YOLO piece detector exported to ONNX (raw head output; decoding + NMS done here)."""

    def __init__(
        self, path: str | Path, conf: float = 0.25, iou: float = 0.7, threads: int | None = None
    ):
        self.sess = make_session(path, threads)
        self.size = _input_size(self.sess)
        self.name = self.sess.get_inputs()[0].name
        self.conf, self.iou = conf, iou

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        """Detect pieces; boxes returned in original-image pixels."""
        x, s, (px, py) = letterbox(img_bgr, self.size)
        out = self.sess.run(None, {self.name: x})[0][0]  # (4+nc, N)
        return decode_detections(
            out, s, px, py, img_bgr.shape[1], img_bgr.shape[0], self.conf, self.iou
        )


def decode_detections(
    out: np.ndarray, scale: float, px: float, py: float, w: int, h: int, conf: float, iou: float
) -> list[Detection]:
    """Decode a raw YOLO head tensor ``(4+nc, N)`` into NMS-filtered detections."""
    out = out.T  # (N, 4+nc)
    scores = out[:, 4:]
    cls = scores.argmax(1)
    confs = scores[np.arange(len(cls)), cls]
    keep = confs >= conf
    if not keep.any():
        return []
    xywh, cls, confs = out[keep, :4], cls[keep], confs[keep]
    x1 = (xywh[:, 0] - xywh[:, 2] / 2 - px) / scale
    y1 = (xywh[:, 1] - xywh[:, 3] / 2 - py) / scale
    x2 = (xywh[:, 0] + xywh[:, 2] / 2 - px) / scale
    y2 = (xywh[:, 1] + xywh[:, 3] / 2 - py) / scale
    x1, x2 = np.clip(x1, 0, w), np.clip(x2, 0, w)
    y1, y2 = np.clip(y1, 0, h), np.clip(y2, 0, h)
    # per-class NMS: offset boxes by class so different classes never suppress each other
    off = cls * 10000.0
    boxes = np.stack([x1 + off, y1, x2 - x1, y2 - y1], 1).tolist()
    idx = cv2.dnn.NMSBoxes(boxes, confs.tolist(), conf, iou)
    idx = np.array(idx).flatten().astype(int)
    return [
        Detection(
            float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), int(cls[i]), float(confs[i])
        )
        for i in idx
    ]


class OnnxCornerModel:
    """YOLO-pose board-corner model exported to ONNX (single class, 4 keypoints)."""

    def __init__(self, path: str | Path, threads: int | None = None):
        self.sess = make_session(path, threads)
        self.size = _input_size(self.sess)
        self.name = self.sess.get_inputs()[0].name

    def find(self, img_bgr: np.ndarray, min_conf: float = 0.05) -> CornerResult | None:
        """Return the most confident board's 4 corners in image pixels, or None."""
        x, s, (px, py) = letterbox(img_bgr, self.size)
        out = self.sess.run(None, {self.name: x})[0][0]  # (5+12, N)
        return decode_corners(out, s, px, py, img_bgr.shape[1], img_bgr.shape[0], min_conf)


def decode_corners(
    out: np.ndarray, scale: float, px: float, py: float, w: int, h: int, min_conf: float = 0.05
) -> CornerResult | None:
    """Decode a raw YOLO-pose head tensor ``(5+12, N)`` into the best board's corners."""
    out = out.T
    i = int(out[:, 4].argmax())
    conf = float(out[i, 4])
    if conf < min_conf:
        return None
    kp = out[i, 5:].reshape(4, 3)[:, :2]
    pts = np.stack([(kp[:, 0] - px) / scale, (kp[:, 1] - py) / scale], 1)
    pts[:, 0], pts[:, 1] = (
        np.clip(pts[:, 0], -w * 0.1, w * 1.1),
        np.clip(pts[:, 1], -h * 0.1, h * 1.1),
    )
    return CornerResult(order_corners(pts), conf)
