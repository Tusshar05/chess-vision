"""Run the Streamlit script headlessly (AppTest) against a mocked API."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

APP = Path(__file__).parent.parent / "services/dashboard/app.py"


def fake_request(method: str, url: str, **kw):
    r = MagicMock()
    r.status_code = 200
    if url.endswith("/health"):
        r.json.return_value = {
            "status": "ok",
            "version": "t",
            "stockfish_available": True,
            "model": {"kind": "smoke", "precision": "fp32"},
        }
    else:
        r.json.return_value = {"total": 0, "items": []}
    return r


def test_dashboard_renders_and_warns_on_smoke_model() -> None:
    with patch("requests.request", side_effect=fake_request):
        at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert any("Smoke-test weights loaded" in e.value for e in at.error)
    assert at.title[0].value.endswith("ChessVision")
