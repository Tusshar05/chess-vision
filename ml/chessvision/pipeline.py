"""End-to-end recognition: photo -> corners -> pieces -> squares -> FEN (+ validation)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import chess
import cv2
import numpy as np

from ml.chessvision import board as B
from ml.chessvision import corners_classic
from ml.chessvision.onnx_infer import CornerResult, OnnxCornerModel, OnnxDetector

WORK_SIDE = 1280  # photos are downscaled once to this long side, matching the training data


def downscale(img_bgr: np.ndarray, max_side: int = WORK_SIDE) -> tuple[np.ndarray, float]:
    """Area-downscale so the long side is <= ``max_side``; returns (image, factor <= 1)."""
    f = min(1.0, max_side / max(img_bgr.shape[:2]))
    if f == 1.0:
        return img_bgr, 1.0
    h, w = img_bgr.shape[:2]
    size = (round(w * f), round(h * f))  # explicit dsize hits OpenCV's fast INTER_AREA path
    return cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA), f


class BoardNotFoundError(RuntimeError):
    """No chessboard could be located in the image."""


class Detector(Protocol):
    """Anything that turns a BGR image into piece detections."""

    def detect(self, img_bgr: np.ndarray) -> list[B.Detection]: ...


class CornerFinder(Protocol):
    """Anything that turns a BGR image into board corners."""

    def find(self, img_bgr: np.ndarray) -> CornerResult | None: ...


@dataclass
class RecognitionResult:
    """Everything the API/dashboard needs from one recognition."""

    fen: str
    turn: str
    rotation: int
    corners: np.ndarray
    corner_source: str  # "learned" | "classic"
    corner_conf: float
    detections: list[B.Detection]
    reconstruction: B.Reconstruction
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def square_confidences(self) -> dict[str, dict]:
        """Per occupied square: piece symbol and detector confidence."""
        return {
            sq: {"piece": r.symbol, "confidence": round(r.conf, 4)}
            for sq, r in sorted(self.reconstruction.squares.items())
        }


class Recognizer:
    """Wires a piece detector and a corner finder into the full photo -> FEN pipeline.

    ``corner_mode``: ``"learned"`` (learned model, classical fallback), ``"learned_only"`` or
    ``"classic"`` (OpenCV baseline only).
    """

    LOW_CONF = 0.4

    def __init__(
        self,
        detector: Detector,
        corner_model: CornerFinder | None = None,
        corner_mode: str = "learned",
        min_corner_conf: float = 0.25,
    ):
        self.detector, self.corner_model = detector, corner_model
        self.corner_mode, self.min_corner_conf = corner_mode, min_corner_conf

    def find_corners(self, img: np.ndarray) -> tuple[CornerResult | None, str]:
        """Apply the configured corner strategy; returns (result, source)."""
        if self.corner_mode != "classic" and self.corner_model is not None:
            res = self.corner_model.find(img)
            if res is not None and res.conf >= self.min_corner_conf:
                return res, "learned"
            if self.corner_mode == "learned_only":
                return res, "learned"
        return corners_classic.find_corners(img), "classic"

    def recognize(
        self, img_bgr: np.ndarray, rotation: int | None = None, turn: str = "w"
    ) -> RecognitionResult:
        """Run the pipeline. ``rotation`` (0-3) overrides auto-orientation; ``turn`` is w/b."""
        t = {}
        t0 = time.perf_counter()
        img_full = img_bgr
        img_bgr, f = downscale(img_full)
        t["resize"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        cres, source = self.find_corners(img_bgr)
        t["corners"] = (time.perf_counter() - t0) * 1000
        if cres is None:
            raise BoardNotFoundError("could not locate a chessboard in the image")
        t0 = time.perf_counter()
        dets = self.detector.detect(img_bgr)
        t["detect"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        if f != 1.0:  # report everything in the coordinates of the original photo
            cres = CornerResult(cres.corners / f, cres.conf)
            dets = [
                B.Detection(d.x1 / f, d.y1 / f, d.x2 / f, d.y2 / f, d.cls, d.conf) for d in dets
            ]
        rec = B.reconstruct(dets, cres.corners, rotation=rotation)
        board = B.build_board(rec, chess.WHITE if turn == "w" else chess.BLACK)
        warnings = B.validate_board(board)
        t["geometry"] = (time.perf_counter() - t0) * 1000
        if source == "classic":
            warnings.append("board corners come from the classical OpenCV fallback (less accurate)")
        if rec.conflicts:
            warnings.append(f"conflicting detections on {', '.join(rec.conflicts)} (kept highest)")
        if rec.dropped_off_board:
            warnings.append(f"{rec.dropped_off_board} detections outside the board were ignored")
        low = [sq for sq, r in rec.squares.items() if r.conf < self.LOW_CONF]
        if low:
            warnings.append(f"low-confidence squares: {', '.join(sorted(low))}")
        if rotation is None:
            _, scores = B.resolve_orientation(B.locate_cells(dets, B.homography(cres.corners))[0])
            ranked = sorted(scores.values(), reverse=True)
            if len(ranked) > 1 and ranked[0] - ranked[1] < 2.0:
                warnings.append("board orientation is ambiguous; set rotation manually if wrong")
        t["total"] = sum(t.values())
        return RecognitionResult(
            board.fen(),
            turn,
            rec.rotation,
            cres.corners,
            source,
            cres.conf,
            dets,
            rec,
            warnings,
            {k: round(v, 1) for k, v in t.items()},
        )


def read_manifest(weights_dir: Path) -> dict:
    """Return ``weights/manifest.json`` (``kind`` is ``smoke`` or ``trained``)."""
    p = weights_dir / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {"kind": "unknown"}


def load_recognizer(
    weights_dir: str | Path | None = None,
    precision: str | None = None,
    conf: float = 0.25,
    corner_mode: str = "learned",
) -> Recognizer:
    """Build a Recognizer from ONNX files in ``weights_dir`` (env ``WEIGHTS_DIR``).

    ``precision``: ``fp32`` or ``int8`` (env ``MODEL_PRECISION``, default fp32).
    """
    wd = Path(weights_dir or os.environ.get("WEIGHTS_DIR", "weights"))
    prec = precision or os.environ.get("MODEL_PRECISION", "fp32")
    suffix = "_int8" if prec == "int8" else ""
    det = OnnxDetector(wd / f"detector{suffix}.onnx", conf=conf)
    cm_path = wd / f"corners{suffix}.onnx"
    cm = OnnxCornerModel(cm_path) if cm_path.exists() else None
    return Recognizer(det, cm, corner_mode)
