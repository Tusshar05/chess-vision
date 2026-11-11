"""Stockfish analysis via python-chess's UCI API (Stockfish is run as a separate binary)."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import chess
import chess.engine


class EngineUnavailableError(RuntimeError):
    """Raised when no Stockfish binary can be found."""


@dataclass
class Line:
    """One principal variation."""

    move: str  # best first move, UCI
    san: str
    score_cp: int | None  # centipawns from White's point of view (None if mate)
    mate: int | None  # moves to mate, White's POV (negative = Black mates)
    pv: list[str]  # SAN moves


@dataclass
class Analysis:
    """Engine analysis of a position."""

    fen: str
    best_move: str | None
    best_move_san: str | None
    score_cp: int | None
    mate: int | None
    lines: list[Line]
    depth: int | None
    game_over: bool = False


def find_stockfish() -> str:
    """Locate the Stockfish binary: $STOCKFISH_PATH, PATH, then common install dirs."""
    for cand in (
        os.environ.get("STOCKFISH_PATH"),
        shutil.which("stockfish"),
        "/usr/games/stockfish",
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
    ):
        if cand and os.path.exists(cand):
            return cand
    raise EngineUnavailableError("Stockfish not found; install it or set STOCKFISH_PATH")


def _line(board: chess.Board, info: chess.engine.InfoDict) -> Line | None:
    pv = info.get("pv")
    if not pv:
        return None
    sc = info["score"].white()
    sans, b = [], board.copy()
    for m in pv[:8]:
        sans.append(b.san(m))
        b.push(m)
    return Line(pv[0].uci(), sans[0], sc.score(), sc.mate(), sans)


def analyse(
    fen: str,
    depth: int = 14,
    time_limit: float | None = None,
    multipv: int = 3,
    engine_path: str | None = None,
) -> Analysis:
    """Analyse ``fen``. Raises ``ValueError`` for an illegal/unparseable position."""
    try:
        board = chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"invalid FEN: {e}") from e
    status = board.status()
    if status & (
        chess.STATUS_NO_WHITE_KING | chess.STATUS_NO_BLACK_KING | chess.STATUS_TOO_MANY_KINGS
    ):
        raise ValueError("invalid position: each side needs exactly one king")
    if board.is_game_over():
        return Analysis(board.fen(), None, None, None, None, [], None, game_over=True)
    limit = chess.engine.Limit(depth=None if time_limit else depth, time=time_limit)
    with chess.engine.SimpleEngine.popen_uci(engine_path or find_stockfish()) as eng:
        infos = eng.analyse(board, limit, multipv=max(1, multipv))
    lines = [ln for i in infos if (ln := _line(board, i))]
    top = lines[0] if lines else None
    return Analysis(
        board.fen(),
        top.move if top else None,
        top.san if top else None,
        top.score_cp if top else None,
        top.mate if top else None,
        lines,
        infos[0].get("depth") if infos else None,
    )
