"""Geometry and position reconstruction: corners -> homography -> squares -> FEN.

Conventions
-----------
* Corners are given in *image space* in clockwise order starting from the top-left
  (``order_corners`` produces this from any 4 points).
* The warped board is a square of ``size`` px; cell ``(col, row)`` with ``(0, 0)`` at the warped
  top-left.
* Orientation ``rot`` in 0..3 says which *board* corner (0=a8, 1=h8, 2=h1, 3=a1, i.e. white at
  the bottom) sits at the canonical image top-left. ``rot=0`` means white is nearest the camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import cv2
import numpy as np

from ml.chessvision.constants import CLASS_TO_SYMBOL, FILES

BOARD_PX = 800
# The base of a piece is a few pixels above the bottom edge of its box (perspective + shadow).
# 0.10 of the box height gave 100% vs 98.7% perfect boards using *ground-truth* boxes and corners
# (see reports/eval/geometry_oracle.json); 0.0 is the plain bottom-centre.
ANCHOR_LIFT = 0.10


@dataclass(frozen=True)
class Detection:
    """One detected piece: xyxy box in image pixels, class id (0..11) and confidence."""

    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    conf: float

    @property
    def anchor(self) -> tuple[float, float]:
        """Bottom-centre of the box, lifted slightly: where the piece stands on the board."""
        return self.anchor_at(ANCHOR_LIFT)

    def anchor_at(self, lift: float) -> tuple[float, float]:
        """Anchor with an explicit lift (fraction of box height above the bottom edge)."""
        return (self.x1 + self.x2) / 2.0, self.y2 - lift * (self.y2 - self.y1)


@dataclass
class SquareResult:
    """The piece assigned to a square."""

    symbol: str
    conf: float
    cls: int


@dataclass
class Reconstruction:
    """Result of mapping detections to a board."""

    squares: dict[str, SquareResult]
    rotation: int
    conflicts: list[str] = field(default_factory=list)
    dropped_off_board: int = 0

    def fen_placement(self) -> str:
        """FEN piece-placement field (first FEN token)."""
        rows = []
        for rank in range(8, 0, -1):
            row, empty = "", 0
            for f in FILES:
                s = self.squares.get(f"{f}{rank}")
                if s is None:
                    empty += 1
                else:
                    row += (str(empty) if empty else "") + s.symbol
                    empty = 0
            rows.append(row + (str(empty) if empty else ""))
        return "/".join(rows)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points clockwise starting at the image top-left (min x+y), returns (4,2) float32."""
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])  # image coords: increasing angle = clockwise
    p = p[np.argsort(ang)]
    start = int(np.argmin(p[:, 0] + p[:, 1]))
    return np.roll(p, -start, axis=0)


def homography(corners: np.ndarray, size: int = BOARD_PX) -> np.ndarray:
    """3x3 homography mapping image pixels to the warped ``size``x``size`` top-down board."""
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    return cv2.getPerspectiveTransform(order_corners(corners), dst)


def project(H: np.ndarray, pt: tuple[float, float]) -> tuple[float, float]:
    """Apply homography ``H`` to one point."""
    v = H @ np.array([pt[0], pt[1], 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def cell_to_square(col: int, row: int, rot: int) -> str:
    """Map a canonical warped cell to an algebraic square for orientation ``rot``."""
    bc, br = [(col, row), (7 - row, col), (7 - col, 7 - row), (row, 7 - col)][rot % 4]
    return f"{FILES[bc]}{8 - br}"


def locate_cells(
    dets: list[Detection],
    H: np.ndarray,
    size: int = BOARD_PX,
    margin: float = 0.2,
    lift: float | None = None,
) -> tuple[list[tuple[int, int, Detection]], int]:
    """Project each detection's anchor to a canonical cell; drop those off the board.

    ``margin`` (in cell units) tolerates anchors that fall just outside the outer edge.
    Returns ``(cells, n_dropped)`` with cells as ``(col, row, detection)``.
    """
    cell = size / 8.0
    out, dropped = [], 0
    for d in dets:
        u, v = project(H, d.anchor if lift is None else d.anchor_at(lift))
        cu, cv_ = u / cell, v / cell
        if cu < -margin or cu > 8 + margin or cv_ < -margin or cv_ > 8 + margin:
            dropped += 1
            continue
        out.append((int(min(max(cu, 0), 7.999)), int(min(max(cv_, 0), 7.999)), d))
    return out, dropped


def build_squares(
    cells: list[tuple[int, int, Detection]], rot: int
) -> tuple[dict[str, SquareResult], list[str]]:
    """Assign detections to squares; on conflict keep the highest-confidence piece."""
    best: dict[str, Detection] = {}
    conflicts: list[str] = []
    for col, row, d in cells:
        sq = cell_to_square(col, row, rot)
        if sq in best:
            conflicts.append(sq)
            if d.conf <= best[sq].conf:
                continue
        best[sq] = d
    return {sq: SquareResult(CLASS_TO_SYMBOL[d.cls], d.conf, d.cls) for sq, d in best.items()}, (
        sorted(set(conflicts))
    )


def orientation_score(squares: dict[str, SquareResult]) -> float:
    """Higher when white pieces sit on low ranks and black on high ranks (white at the bottom).

    Kings and pawns are weighted more: they are the most reliable cue in any phase of the game.
    """
    score = 0.0
    for sq, s in squares.items():
        rank = int(sq[1]) - 4.5
        w = 2.0 if s.symbol.lower() in "kp" else 1.0
        score += -rank * w if s.symbol.isupper() else rank * w
    return score


def resolve_orientation(cells: list[tuple[int, int, Detection]]) -> tuple[int, dict[int, float]]:
    """Pick the rotation (0..3) under which the piece layout looks like white-at-the-bottom."""
    scores = {r: orientation_score(build_squares(cells, r)[0]) for r in range(4)}
    return max(scores, key=lambda r: scores[r]), scores


def reconstruct(
    dets: list[Detection],
    corners: np.ndarray,
    rotation: int | None = None,
    size: int = BOARD_PX,
    lift: float | None = None,
) -> Reconstruction:
    """Full geometry stage. ``rotation=None`` auto-detects board orientation from the pieces."""
    H = homography(corners, size)
    cells, dropped = locate_cells(dets, H, size, lift=lift)
    rot = resolve_orientation(cells)[0] if rotation is None else rotation
    squares, conflicts = build_squares(cells, rot)
    return Reconstruction(squares, rot, conflicts, dropped)


def build_board(rec: Reconstruction, turn: chess.Color = chess.WHITE) -> chess.Board:
    """python-chess Board from a reconstruction; castling rights inferred from home squares."""
    board = chess.Board(None)
    for sq, s in rec.squares.items():
        board.set_piece_at(chess.parse_square(sq), chess.Piece.from_symbol(s.symbol))
    board.turn = turn
    board.castling_rights = chess.BB_EMPTY
    for color, king_sq, rooks in (
        (chess.WHITE, chess.E1, (chess.H1, chess.A1)),
        (chess.BLACK, chess.E8, (chess.H8, chess.A8)),
    ):
        if board.piece_at(king_sq) == chess.Piece(chess.KING, color):
            for r in rooks:
                if board.piece_at(r) == chess.Piece(chess.ROOK, color):
                    board.castling_rights |= chess.BB_SQUARES[r]
    return board


def validate_board(board: chess.Board) -> list[str]:
    """Sanity checks on a recognised position; returns human-readable warnings."""
    w: list[str] = []
    for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        kings = len(board.pieces(chess.KING, color))
        if kings != 1:
            w.append(f"{name} has {kings} kings (expected exactly 1)")
        if len(board.pieces(chess.PAWN, color)) > 8:
            w.append(f"{name} has more than 8 pawns")
        n = chess.popcount(board.occupied_co[color])
        if n > 16:
            w.append(f"{name} has {n} pieces (max 16)")
    back = chess.BB_RANK_1 | chess.BB_RANK_8
    if board.pawns & back:
        w.append("pawn on rank 1 or 8")
    if (
        len(board.pieces(chess.KING, chess.WHITE)) == 1
        and len(board.pieces(chess.KING, chess.BLACK)) == 1
    ):
        status = board.status()
        if status & chess.STATUS_OPPOSITE_CHECK:
            w.append("side not to move is in check (wrong side to move or misread piece)")
        if status & chess.STATUS_TOO_MANY_CHECKERS:
            w.append("impossible number of checkers")
    if chess.popcount(board.occupied) < 2:
        w.append("fewer than two pieces detected")
    return w
