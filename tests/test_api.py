"""API endpoint tests with mocked model sessions and Stockfish."""

import json

import cv2
import numpy as np

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_health_reports_model_and_engine(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model"]["kind"] == "unknown"  # tmp weights dir has no manifest
    assert {"status", "version", "stockfish_available"} <= set(body)


def test_recognize_returns_fen_squares_and_images(client, photo_bytes) -> None:
    r = client.post("/v1/recognize", files={"file": ("p.png", photo_bytes, "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["fen"] == START
    assert d["squares"]["e1"] == {"piece": "K", "confidence": 0.95}
    assert d["warnings"] == [] and d["corner_source"] == "learned"
    assert len(d["detections"]) == 32
    assert d["annotated_image"] and d["board_image"] and d["board_svg"].startswith("<svg")
    jpg = np.frombuffer(__import__("base64").b64decode(d["board_image"]), np.uint8)
    assert cv2.imdecode(jpg, cv2.IMREAD_COLOR).shape[:2] == (640, 640)


def test_recognize_without_images_and_turn(client, photo_bytes) -> None:
    r = client.post(
        "/v1/recognize",
        data={"turn": "b", "include_images": "false"},
        files={"file": ("p.png", photo_bytes, "image/png")},
    )
    d = r.json()
    assert " b " in d["fen"] and d["annotated_image"] is None


def test_rejects_wrong_content_type(client) -> None:
    r = client.post("/v1/recognize", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_media_type"


def test_rejects_corrupt_image(client) -> None:
    r = client.post("/v1/recognize", files={"file": ("a.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_image"


def test_rejects_oversized_upload(client) -> None:
    big = b"\xff\xd8" + b"0" * (1024 * 1024 + 10)
    r = client.post("/v1/recognize", files={"file": ("a.jpg", big, "image/jpeg")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_rejects_bad_rotation_and_turn(client, photo_bytes) -> None:
    f = {"file": ("p.png", photo_bytes, "image/png")}
    assert client.post("/v1/recognize", data={"rotation": "7"}, files=f).status_code == 422
    assert client.post("/v1/recognize", data={"turn": "x"}, files=f).status_code == 422


def test_missing_file_is_validation_error(client) -> None:
    r = client.post("/v1/recognize")
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_analyze_ok(client) -> None:
    r = client.post("/v1/analyze", json={"fen": START, "depth": 8, "multipv": 2})
    assert r.status_code == 200
    d = r.json()
    assert d["best_move"] == "e2e4" and len(d["lines"]) == 2 and d["score_cp"] == 25


def test_analyze_invalid_fen(client) -> None:
    r = client.post("/v1/analyze", json={"fen": "this is not a fen at all"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_position"


def test_analyze_depth_bounds(client) -> None:
    assert client.post("/v1/analyze", json={"fen": START, "depth": 99}).status_code == 422


def test_recognize_and_analyze(client, photo_bytes) -> None:
    r = client.post(
        "/v1/recognize-and-analyze",
        data={"depth": "6", "include_images": "false"},
        files={"file": ("p.png", photo_bytes, "image/png")},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["fen"] == START and d["analysis"]["best_move_san"] == "e4"
    assert d["analysis_error"] is None


def test_history_records_recognitions(client, photo_bytes) -> None:
    for _ in range(3):
        client.post(
            "/v1/recognize",
            data={"include_images": "false"},
            files={"file": ("shot.png", photo_bytes, "image/png")},
        )
    r = client.get("/v1/history", params={"limit": 2})
    d = r.json()
    assert d["total"] == 3 and len(d["items"]) == 2
    assert d["items"][0]["id"] > d["items"][1]["id"] and d["items"][0]["filename"] == "shot.png"
    assert d["items"][0]["fen"] == START


def test_metrics_endpoint(client, photo_bytes) -> None:
    client.post("/v1/recognize", files={"file": ("p.png", photo_bytes, "image/png")})
    text = client.get("/metrics").text
    assert "chessvision_requests_total" in text and "chessvision_recognitions_total" in text


def test_openapi_lists_endpoints(client) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for p in (
        "/v1/recognize",
        "/v1/analyze",
        "/v1/recognize-and-analyze",
        "/v1/history",
        "/health",
    ):
        assert p in paths


def test_request_id_header_roundtrip(client) -> None:
    r = client.get("/health", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"


def test_models_unavailable_returns_503(tmp_path, photo_bytes) -> None:
    from fastapi.testclient import TestClient

    from services.api.app.config import Settings
    from services.api.app.main import create_app

    cfg = Settings(weights_dir=str(tmp_path / "nothing"), db_url=f"sqlite:///{tmp_path / 'x.db'}")
    with TestClient(create_app(cfg)) as c:
        r = c.post("/v1/recognize", files={"file": ("p.png", photo_bytes, "image/png")})
        assert r.status_code == 503 and r.json()["error"]["code"] == "models_unavailable"
        assert c.get("/health").json()["status"] == "degraded"
        json.dumps(c.get("/health").json())
