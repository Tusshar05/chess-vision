"""Shared fixtures: an API app wired to fake models and a fake engine (no weights, no Stockfish)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml.chessvision import board as B
from ml.chessvision import engine as sf
from ml.chessvision.onnx_infer import CornerResult
from ml.chessvision.pipeline import Recognizer
from services.api.app.config import Settings
from services.api.app.main import create_app

CORNERS = np.array([[100, 50], [900, 50], [900, 850], [100, 850]], dtype=np.float32)


class FakeDetector:
    """Start position as seen through ``CORNERS``."""

    def detect(self, img_bgr: np.ndarray) -> list[B.Detection]:
        back = [1, 2, 3, 4, 5, 3, 2, 1]
        out = []
        for c in range(8):
            for row, cls in ((7, back[c]), (6, 0), (1, 6), (0, 6 + back[c])):
                cx, cy = 100 + c * 100 + 50, 50 + row * 100 + 50
                out.append(B.Detection(cx - 30, cy - 120, cx + 30, cy + 12, cls, 0.95))
        return out


class FakeCorners:
    """Always finds ``CORNERS``."""

    def find(self, img_bgr: np.ndarray) -> CornerResult:
        return CornerResult(CORNERS, 0.9)


def fake_analyse(fen: str, depth: int = 14, time_limit=None, multipv: int = 3, **_):
    """Stand-in for Stockfish: validates the FEN like the real wrapper, returns canned lines."""
    import chess

    try:
        b = chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"invalid FEN: {e}") from e
    if len(b.pieces(chess.KING, chess.WHITE)) != 1 or len(b.pieces(chess.KING, chess.BLACK)) != 1:
        raise ValueError("invalid position: each side needs exactly one king")
    line = sf.Line("e2e4", "e4", 25, None, ["e4", "e5"])
    return sf.Analysis(b.fen(), "e2e4", "e4", 25, None, [line] * multipv, depth)


@pytest.fixture()
def photo_bytes() -> bytes:
    """A small valid PNG (content is irrelevant to the fake models)."""
    ok, buf = cv2.imencode(".png", np.full((900, 1000, 3), 120, np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture()
def client(tmp_path):
    """TestClient on an app using fake models, a fake engine and a temp SQLite DB."""
    cfg = Settings(
        weights_dir=str(tmp_path), db_url=f"sqlite:///{tmp_path / 'h.db'}", max_upload_mb=1
    )
    rec = Recognizer(FakeDetector(), FakeCorners())
    app = create_app(cfg, recognizer=rec, analyse_fn=fake_analyse)
    with TestClient(app) as c:
        yield c
