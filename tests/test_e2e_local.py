"""Local end-to-end test with real model files (skipped in CI where weights are absent)."""

import json
from pathlib import Path

import chess
import cv2
import pytest

from ml.chessvision.pipeline import load_recognizer

WEIGHTS = Path("weights")
SAMPLES = Path("samples")
needs_models = pytest.mark.skipif(
    not (WEIGHTS / "detector.onnx").exists() or not (SAMPLES / "labels.json").exists(),
    reason="exported model files / samples not present",
)


@pytest.mark.e2e
@needs_models
def test_real_models_on_sample_photo() -> None:
    labels = json.loads((SAMPLES / "labels.json").read_text())
    name = sorted(labels)[0]
    img = cv2.imread(str(SAMPLES / name))
    res = load_recognizer(WEIGHTS, conf=0.05).recognize(img)
    chess.Board(res.fen)  # the produced FEN must always parse
    assert res.corners.shape == (4, 2)
    assert set(res.timings_ms) >= {"corners", "detect", "geometry", "total"}


@pytest.mark.e2e
@needs_models
def test_real_models_via_api(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from services.api.app.config import Settings
    from services.api.app.main import create_app

    name = sorted(json.loads((SAMPLES / "labels.json").read_text()))[0]
    cfg = Settings(weights_dir=str(WEIGHTS), db_url=f"sqlite:///{tmp_path / 'e.db'}")
    with TestClient(create_app(cfg)) as c:
        data = (SAMPLES / name).read_bytes()
        r = c.post(
            "/v1/recognize-and-analyze",
            files={"file": (name, data, "image/jpeg")},
            data={"include_images": "false", "depth": "6"},
        )
        assert r.status_code == 200
        assert "fen" in r.json()
