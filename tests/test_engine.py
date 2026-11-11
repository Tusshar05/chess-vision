"""Stockfish wrapper tests (skipped if no Stockfish binary is installed)."""

import pytest

from ml.chessvision.engine import EngineUnavailableError, analyse, find_stockfish

try:
    find_stockfish()
    HAVE_SF = True
except EngineUnavailableError:
    HAVE_SF = False

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_invalid_fen_raises() -> None:
    with pytest.raises(ValueError):
        analyse("not a fen")


def test_missing_king_rejected_before_engine() -> None:
    with pytest.raises(ValueError):
        analyse("8/8/8/8/8/8/8/K7 w - - 0 1")


def test_game_over_short_circuits() -> None:
    mated = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert analyse(mated).game_over


@pytest.mark.skipif(not HAVE_SF, reason="stockfish not installed")
def test_analysis_start_position() -> None:
    a = analyse(START, depth=8, multipv=3)
    assert a.best_move and len(a.lines) == 3
