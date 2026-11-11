"""Pure helpers for the dashboard (unit-tested; no Streamlit import)."""

from __future__ import annotations

import base64
from urllib.parse import quote

import chess
import chess.svg


def lichess_url(fen: str) -> str:
    """Lichess analysis-board URL for a FEN (spaces become underscores)."""
    return "https://lichess.org/analysis/standard/" + quote(fen.replace(" ", "_"), safe="/_-")


def fen_error(fen: str) -> str | None:
    """None if ``fen`` is a legal, analysable position, else a short reason."""
    try:
        b = chess.Board(fen)
    except ValueError as e:
        return f"not a valid FEN ({e})"
    if not b.is_valid():
        problems = [
            s.name.replace("STATUS_", "").replace("_", " ").lower()
            for s in chess.Status
            if s in b.status() and s != chess.STATUS_VALID
        ]
        return "illegal position: " + ", ".join(problems)
    return None


def svg_data_uri(
    fen: str, best_move: str | None = None, flipped: bool = False, size: int = 440
) -> str:
    """``data:`` URI of the board SVG, optionally with an arrow for ``best_move``."""
    b = chess.Board(fen)
    arrows = []
    if best_move:
        m = chess.Move.from_uci(best_move)
        arrows = [chess.svg.Arrow(m.from_square, m.to_square)]
    svg = chess.svg.board(b, arrows=arrows, flipped=flipped, size=size)
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def eval_text(score_cp: int | None, mate: int | None) -> str:
    """Human-readable evaluation from White's point of view."""
    if mate is not None:
        return f"mate in {abs(mate)} for {'White' if mate > 0 else 'Black'}"
    if score_cp is None:
        return "n/a"
    return f"{score_cp / 100:+.2f}"


def set_turn(fen: str, turn: str) -> str:
    """Return ``fen`` with the side to move replaced (``w`` or ``b``)."""
    parts = fen.split()
    parts[1] = turn
    return " ".join(parts)
