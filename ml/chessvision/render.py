"""Drawing helpers: annotated photo, top-down warped board, SVG board."""

from __future__ import annotations

import chess
import chess.svg
import cv2
import numpy as np

from ml.chessvision import board as B
from ml.chessvision.constants import CLASS_NAMES, FILES

WHITE_COL, BLACK_COL = (60, 200, 60), (60, 60, 230)


def annotate(
    img_bgr: np.ndarray, corners: np.ndarray | None, dets: list[B.Detection]
) -> np.ndarray:
    """Copy of the photo with piece boxes, labels and the board outline drawn on it."""
    out = img_bgr.copy()
    t = max(2, round(max(out.shape[:2]) / 600))
    if corners is not None:
        cv2.polylines(out, [corners.astype(np.int32)], True, (0, 220, 255), t + 1)
    for d in dets:
        col = WHITE_COL if d.cls < 6 else BLACK_COL
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), col, t)
        label = f"{CLASS_NAMES[d.cls].split('-')[1][:2]} {d.conf:.2f}"
        cv2.putText(
            out,
            label,
            (int(d.x1), max(12, int(d.y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * t / 2 + 0.3,
            col,
            max(1, t - 1),
            cv2.LINE_AA,
        )
    return out


def top_down(
    img_bgr: np.ndarray, corners: np.ndarray, rec: B.Reconstruction, size: int = 640
) -> np.ndarray:
    """Warp the photo to a top-down board, rotate white to the bottom, draw grid + pieces."""
    H = B.homography(corners, size)
    warped = cv2.warpPerspective(img_bgr, H, (size, size))
    warped = np.ascontiguousarray(np.rot90(warped, k=-rec.rotation))  # clockwise quarter turns
    cell = size // 8
    for i in range(9):
        cv2.line(warped, (i * cell, 0), (i * cell, size), (255, 255, 255), 1)
        cv2.line(warped, (0, i * cell), (size, i * cell), (255, 255, 255), 1)
    for sq, r in rec.squares.items():
        c, rk = FILES.index(sq[0]), 8 - int(sq[1])
        cx, cy = c * cell + cell // 2, rk * cell + cell // 2
        cv2.circle(
            warped, (cx, cy), cell // 3, (255, 255, 255) if r.symbol.isupper() else (0, 0, 0), -1
        )
        cv2.circle(warped, (cx, cy), cell // 3, (0, 140, 255), 2)
        cv2.putText(
            warped,
            r.symbol,
            (cx - 9, cy + 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0) if r.symbol.isupper() else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    for i in range(8):
        cv2.putText(
            warped,
            FILES[i],
            (i * cell + 3, size - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            warped,
            str(8 - i),
            (3, i * cell + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return warped


def board_svg(fen: str, last_move: str | None = None, size: int = 420) -> str:
    """SVG of a position (python-chess), optionally highlighting an engine move."""
    b = chess.Board(fen)
    arrows = []
    if last_move:
        m = chess.Move.from_uci(last_move)
        arrows = [chess.svg.Arrow(m.from_square, m.to_square)]
    return chess.svg.board(b, arrows=arrows, size=size)
