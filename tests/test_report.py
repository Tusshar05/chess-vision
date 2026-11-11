"""make_report: smoke output must never reach the README; trained output renders tables."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "make_report", Path(__file__).parent.parent / "scripts/make_report.py"
)
mr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr)

E2E = {
    "boards_0_errors_pct": 40.0,
    "boards_le1_errors_pct": 60.0,
    "mean_errors_per_board": 1.5,
    "square_accuracy_pct": 97.7,
}


def _write(root: Path, kind: str) -> None:
    (root / "eval").mkdir(parents=True)
    (root / "eval/WEIGHTS.json").write_text(
        json.dumps({"kind": kind, "detector": "d", "corners": "c"})
    )
    (root / "eval/detector_metrics.json").write_text(
        json.dumps({"mAP50": 0.9, "mAP50-95": 0.7, "precision": 0.8, "recall": 0.85})
    )
    (root / "eval/e2e_summary.json").write_text(
        json.dumps(
            {
                "end_to_end_auto_orientation": E2E,
                "end_to_end_given_orientation": E2E,
                "end_to_end_classic_corners_given_orientation": E2E,
                "paper_reference": {"board_accuracy_pct": 15.26, "note": "indicative."},
            }
        )
    )
    cs = {
        "found_pct": 100,
        "median_err_pct_of_diag": 0.5,
        "mean_err_pct_of_diag": 1.0,
        "within_2pct_diag": 90,
        "within_5pct_diag": 95,
    }
    (root / "eval/corner_comparison.json").write_text(
        json.dumps({"learned_with_fallback": cs, "classic_opencv": cs})
    )
    (root / "benchmark.json").write_text(
        json.dumps(
            {
                "weights_kind": kind,
                "machine": {"chip": "M", "cpu_cores_physical": 8, "ram_gb": 16},
                "backends": {
                    "ONNX FP32": {
                        "p50_ms": 50,
                        "p95_ms": 60,
                        "boards_0_errors_pct": 40,
                        "mean_errors_per_board": 1.5,
                    }
                },
            }
        )
    )


def test_smoke_weights_show_pending(tmp_path, monkeypatch) -> None:
    _write(tmp_path, "smoke")
    monkeypatch.setattr(mr, "R", tmp_path)
    assert mr.results_block() == mr.PENDING and mr.bench_block() == mr.PENDING


def test_trained_weights_render_tables(tmp_path, monkeypatch) -> None:
    _write(tmp_path, "trained")
    monkeypatch.setattr(mr, "R", tmp_path)
    out = mr.results_block()
    assert "| 0.9 | 0.7 | 0.8 | 0.85 |" in out and "15.26%" in out and "OpenCV baseline" in out
    assert "ONNX FP32 | 50 | 60" in mr.bench_block()


def test_main_replaces_blocks(tmp_path, monkeypatch) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("a\n<!-- BENCH:START -->\nold\n<!-- BENCH:END -->\nb\n")
    monkeypatch.setattr(mr, "README", readme)
    monkeypatch.setattr(mr, "R", tmp_path)
    mr.main()
    assert "old" not in readme.read_text() and "Pending" in readme.read_text()
