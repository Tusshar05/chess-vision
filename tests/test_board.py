"""Unit tests for the geometry / FEN logic using synthetic boxes."""

import chess
import numpy as np
import pytest

from ml.chessvision import board as B
from ml.chessvision.board import Detection, reconstruct

# Image-space board: a 800x800 square at offset (100, 50) so the homography is a pure translation.
CORNERS = np.array([[100, 50], [900, 50], [900, 850], [100, 850]], dtype=np.float32)


def det_on(col: int, row: int, cls: int, conf: float = 0.9) -> Detection:
    """A box whose base sits at the centre of canonical cell (col,row) of CORNERS."""
    cx, cy = 100 + col * 100 + 50, 50 + row * 100 + 50
    return Detection(cx - 30, cy - 120, cx + 30, cy + 12, cls, conf)  # lift keeps anchor in cell


def test_order_corners_any_permutation() -> None:
    shuffled = CORNERS[[2, 0, 3, 1]]
    assert np.allclose(B.order_corners(shuffled), CORNERS)


def test_homography_maps_corners_to_square() -> None:
    H = B.homography(CORNERS, 800)
    assert B.project(H, (100, 50)) == pytest.approx((0, 0), abs=1e-3)
    assert B.project(H, (900, 850)) == pytest.approx((800, 800), abs=1e-3)
    assert B.project(H, (500, 450)) == pytest.approx((400, 400), abs=1e-3)


def test_homography_perspective_trapezoid() -> None:
    trap = np.array([[300, 100], [700, 100], [900, 500], [100, 500]], dtype=np.float32)
    H = B.homography(trap, 800)
    # the intersection of the diagonals maps to the board centre
    inter = np.linalg.solve(
        np.array(
            [
                [trap[2][0] - trap[0][0], -(trap[3][0] - trap[1][0])],
                [trap[2][1] - trap[0][1], -(trap[3][1] - trap[1][1])],
            ]
        ),
        trap[1] - trap[0],
    )
    centre = trap[0] + inter[0] * (trap[2] - trap[0])
    assert B.project(H, tuple(centre)) == pytest.approx((400, 400), abs=1e-2)


@pytest.mark.parametrize("rot,sq", [(0, "a8"), (1, "h8"), (2, "h1"), (3, "a1")])
def test_cell_to_square_top_left_per_rotation(rot: int, sq: str) -> None:
    assert B.cell_to_square(0, 0, rot) == sq


def test_cell_to_square_rotations_are_bijections() -> None:
    for rot in range(4):
        squares = {B.cell_to_square(c, r, rot) for c in range(8) for r in range(8)}
        assert len(squares) == 64


def test_square_assignment_white_at_bottom() -> None:
    # white king (cls 5) on e1 -> canonical col 4,row 7 ; black king (cls 11) on e8 -> col 4,row 0
    rec = reconstruct([det_on(4, 7, 5), det_on(4, 0, 11)], CORNERS, rotation=0)
    assert rec.squares["e1"].symbol == "K"
    assert rec.squares["e8"].symbol == "k"


def test_conflict_keeps_highest_confidence() -> None:
    lo, hi = det_on(3, 3, 0, conf=0.4), det_on(3, 3, 1, conf=0.8)
    rec = reconstruct([lo, hi], CORNERS, rotation=0)
    assert rec.squares["d5"].symbol == "R"
    assert rec.conflicts == ["d5"]


def test_off_board_pieces_are_dropped() -> None:
    off = Detection(1000, 100, 1060, 200, 0, 0.9)  # captured piece beside the board
    rec = reconstruct([det_on(0, 0, 7), off], CORNERS, rotation=0)
    assert rec.dropped_off_board == 1
    assert len(rec.squares) == 1


def start_position_dets(white_row_bottom: bool = True) -> list[Detection]:
    dets = []
    back = [1, 2, 3, 4, 5, 3, 2, 1]  # rook knight bishop queen king bishop knight rook (white ids)
    for c in range(8):
        dets += [det_on(c, 7, back[c]), det_on(c, 6, 0), det_on(c, 1, 6), det_on(c, 0, 6 + back[c])]
    return dets


def test_full_start_position_fen() -> None:
    rec = reconstruct(start_position_dets(), CORNERS, rotation=0)
    assert rec.fen_placement() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


@pytest.mark.parametrize("rot", range(4))
def test_orientation_resolution_rotated_board(rot: int) -> None:
    """Rotate the synthetic scene and check auto-orientation recovers the same position."""
    rotated = []
    for d in start_position_dets():
        cx, cy = (d.x1 + d.x2) / 2, d.y2 - B.ANCHOR_LIFT * (d.y2 - d.y1)
        col, row = int((cx - 100) // 100), int((cy - 50) // 100)
        for _ in range(rot):  # inverse of the quarter-turn used by cell_to_square
            col, row = row, 7 - col
        rotated.append(det_on(col, row, d.cls))
    rec = reconstruct(rotated, CORNERS)
    assert rec.fen_placement() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    assert rec.rotation == rot


def test_build_board_and_validate_ok() -> None:
    rec = reconstruct(start_position_dets(), CORNERS, rotation=0)
    board = B.build_board(rec)
    assert B.validate_board(board) == []
    assert board.fen().startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq")


def test_validate_flags_missing_king_and_bad_pawn() -> None:
    board = chess.Board(None)
    board.set_piece_at(chess.A1, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    warnings = B.validate_board(board)
    assert any("white has 0 kings" in w for w in warnings)
    assert any("pawn on rank 1 or 8" in w for w in warnings)


def test_validate_flags_too_many_pieces() -> None:
    board = chess.Board(None)
    for i in range(17):
        board.set_piece_at(i + 8 * 2 if i < 16 else 40, chess.Piece(chess.KNIGHT, chess.WHITE))
    assert any("pieces (max 16)" in w for w in B.validate_board(board))


def test_castling_rights_inferred_only_with_home_pieces() -> None:
    rec = reconstruct([det_on(4, 7, 5), det_on(7, 7, 1), det_on(4, 0, 11)], CORNERS, rotation=0)
    board = B.build_board(rec)
    assert board.has_kingside_castling_rights(chess.WHITE)
    assert not board.has_queenside_castling_rights(chess.WHITE)
    assert not board.has_kingside_castling_rights(chess.BLACK)
