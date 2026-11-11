"""Tests for decoding, the pipeline (with fake models) and rendering."""

import numpy as np
import pytest

from ml.chessvision import board as B
from ml.chessvision import render
from ml.chessvision.onnx_infer import CornerResult, decode_corners, decode_detections, letterbox
from ml.chessvision.pipeline import BoardNotFoundError, Recognizer

CORNERS = np.array([[100, 50], [900, 50], [900, 850], [100, 850]], dtype=np.float32)


def test_letterbox_shapes_and_padding() -> None:
    img = np.zeros((200, 400, 3), np.uint8)
    x, s, (px, py) = letterbox(img, 640)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert s == pytest.approx(1.6) and px == 0 and py == 160


def test_decode_detections_nms_and_rescale() -> None:
    nc = 12
    out = np.zeros((4 + nc, 3), np.float32)
    # two overlapping boxes of class 2 (keep the stronger), one of class 5
    out[:4, 0] = [100, 100, 50, 80]
    out[4 + 2, 0] = 0.9
    out[:4, 1] = [102, 101, 50, 80]
    out[4 + 2, 1] = 0.6
    out[:4, 2] = [300, 300, 40, 60]
    out[4 + 5, 2] = 0.8
    dets = decode_detections(out, scale=0.5, px=0, py=0, w=1000, h=1000, conf=0.25, iou=0.7)
    assert sorted(d.cls for d in dets) == [2, 5]
    d2 = next(d for d in dets if d.cls == 2)
    assert d2.conf == pytest.approx(0.9) and d2.x1 == pytest.approx(150.0)  # (100-25)/0.5


def test_decode_detections_empty() -> None:
    assert decode_detections(np.zeros((16, 5), np.float32), 1, 0, 0, 10, 10, 0.25, 0.7) == []


def test_decode_corners_picks_best_and_orders() -> None:
    out = np.zeros((17, 4), np.float32)
    out[4, 2] = 0.9
    kps = [(500, 100), (100, 100), (100, 500), (500, 500)]  # shuffled order
    out[5:, 2] = np.array([[x, y, 1.0] for x, y in kps], np.float32).ravel()
    res = decode_corners(out, scale=1.0, px=0, py=0, w=640, h=640)
    assert res is not None and res.conf == pytest.approx(0.9)
    assert res.corners[0].tolist() == [100, 100]  # TL first
    assert decode_corners(np.zeros((17, 4), np.float32), 1, 0, 0, 10, 10) is None


class FakeDetector:
    """Returns a fixed start position seen through CORNERS."""

    def detect(self, img_bgr: np.ndarray) -> list[B.Detection]:
        back = [1, 2, 3, 4, 5, 3, 2, 1]
        out = []
        for c in range(8):
            for row, cls in ((7, back[c]), (6, 0), (1, 6), (0, 6 + back[c])):
                cx, cy = 100 + c * 100 + 50, 50 + row * 100 + 50
                out.append(B.Detection(cx - 30, cy - 120, cx + 30, cy + 12, cls, 0.95))
        return out


class FakeCorners:
    def __init__(self, res: CornerResult | None):
        self.res = res

    def find(self, img_bgr: np.ndarray) -> CornerResult | None:
        return self.res


def test_pipeline_end_to_end_with_fakes() -> None:
    img = np.zeros((900, 1000, 3), np.uint8)
    rec = Recognizer(FakeDetector(), FakeCorners(CornerResult(CORNERS, 0.9)))
    res = rec.recognize(img)
    assert res.fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert res.warnings == [] and res.corner_source == "learned"
    assert res.square_confidences()["e1"]["piece"] == "K"
    assert set(res.timings_ms) == {"resize", "corners", "detect", "geometry", "total"}


def test_pipeline_turn_black() -> None:
    rec = Recognizer(FakeDetector(), FakeCorners(CornerResult(CORNERS, 0.9)))
    res = rec.recognize(np.zeros((900, 1000, 3), np.uint8), turn="b")
    assert " b " in res.fen


def test_pipeline_board_not_found() -> None:
    rec = Recognizer(FakeDetector(), FakeCorners(None), corner_mode="learned_only")
    with pytest.raises(BoardNotFoundError):
        rec.recognize(np.zeros((100, 100, 3), np.uint8))


def test_render_outputs() -> None:
    img = np.full((900, 1000, 3), 127, np.uint8)
    rec = Recognizer(FakeDetector(), FakeCorners(CornerResult(CORNERS, 0.9)))
    res = rec.recognize(img)
    assert render.annotate(img, res.corners, res.detections).shape == img.shape
    assert render.top_down(img, res.corners, res.reconstruction).shape == (640, 640, 3)
    svg = render.board_svg(res.fen, "e2e4")
    assert svg.startswith("<svg")


def test_downscale_keeps_original_coordinates() -> None:
    from ml.chessvision.pipeline import downscale

    big = np.zeros((2000, 4000, 3), np.uint8)
    small, f = downscale(big)
    assert small.shape[1] == 1280 and f == pytest.approx(0.32)
    same, f1 = downscale(np.zeros((500, 600, 3), np.uint8))
    assert f1 == 1.0 and same.shape[:2] == (500, 600)
