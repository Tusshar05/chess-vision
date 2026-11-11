"""Dashboard helper tests."""

from services.dashboard import helpers as H

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_lichess_url() -> None:
    u = H.lichess_url(START)
    assert u.startswith("https://lichess.org/analysis/standard/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/")
    assert " " not in u and u.endswith("w_KQkq_-_0_1")


def test_fen_error() -> None:
    assert H.fen_error(START) is None
    assert "not a valid FEN" in H.fen_error("garbage")
    assert "illegal position" in H.fen_error("8/8/8/8/8/8/8/K7 w - - 0 1")


def test_eval_text() -> None:
    assert H.eval_text(25, None) == "+0.25"
    assert H.eval_text(-130, None) == "-1.30"
    assert H.eval_text(None, -3) == "mate in 3 for Black"
    assert H.eval_text(None, None) == "n/a"


def test_set_turn_and_svg() -> None:
    assert H.set_turn(START, "b").split()[1] == "b"
    assert H.svg_data_uri(START, "e2e4").startswith("data:image/svg+xml;base64,")
