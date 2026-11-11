"""Classical (no learning) board-corner baseline using OpenCV.

Strategy: (1) look for the 7x7 inner chessboard corners with ``findChessboardCornersSB`` and
extrapolate to the outer corners; (2) otherwise find the largest quadrilateral in the edge map
(the printed board mat) and shrink it to the 8x8 playing area using a margin measured on *training*
annotations (``MARGIN_UV``). Intended as a comparison baseline for the learned corner model.
"""

from __future__ import annotations

import cv2
import numpy as np

from ml.chessvision.board import order_corners
from ml.chessvision.onnx_infer import CornerResult

# (u, v) of the playing-area corners inside the mat quad (unit square), measured on 36 train images
# by ``scripts/calibrate_classic_corners.py``; the mat is slightly larger than the 8x8 grid.
MARGIN_UV = np.array(
    [[0.079, 0.079], [0.921, 0.079], [0.921, 0.921], [0.079, 0.921]], dtype=np.float32
)
WORK = 960


def _inner_corners(gray: np.ndarray) -> np.ndarray | None:
    """Outer 8x8 corners via the 7x7 inner-corner detector, or None if it fails."""
    ok, pts = cv2.findChessboardCornersSB(gray, (7, 7), cv2.CALIB_CB_EXHAUSTIVE)
    if not ok:
        return None
    grid = pts.reshape(7, 7, 2)
    src = np.array([[0, 0], [6, 0], [6, 6], [0, 6]], np.float32) + 1  # inner grid -> board coords
    dst = np.array([grid[0, 0], grid[0, 6], grid[6, 6], grid[6, 0]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    outer = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(outer, H).reshape(4, 2)


def _largest_quad(gray: np.ndarray) -> np.ndarray | None:
    """Largest convex quadrilateral in the edge map, or None."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_img = gray.shape[0] * gray.shape[1]
    best, best_area = None, 0.08 * area_img
    for c in cnts:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if area < best_area or area > 0.98 * area_img:
            continue
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.03, 0.05, 0.08):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                best, best_area = approx.reshape(4, 2).astype(np.float32), area
                break
    return best


def find_corners(img_bgr: np.ndarray, margin_uv: np.ndarray = MARGIN_UV) -> CornerResult | None:
    """Locate the playing-area corners (image pixels, clockwise from top-left) or None."""
    h, w = img_bgr.shape[:2]
    s = min(1.0, WORK / max(h, w))
    small = (
        cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    pts = _inner_corners(gray)
    conf = 0.9
    if pts is None:
        quad = _largest_quad(gray)
        if quad is None:
            return None
        quad = order_corners(quad)
        H = cv2.getPerspectiveTransform(
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32), quad
        )
        pts = cv2.perspectiveTransform(margin_uv.reshape(-1, 1, 2), H).reshape(4, 2)
        conf = 0.5
    return CornerResult(order_corners(pts / s), conf)
