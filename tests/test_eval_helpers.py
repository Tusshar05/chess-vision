"""Evaluation helpers, report assembly and the classical corner baseline."""

import json

import cv2
import numpy as np
import pytest

from ml import colab_runner, evaluate, training_runs
from ml.chessvision import corners_classic


def test_board_errors_counts_all_difference_types() -> None:
    gt = {"e1": "K", "e8": "k", "a2": "P"}
    assert evaluate.board_errors(gt, dict(gt)) == 0
    assert evaluate.board_errors(gt, {"e1": "K", "e8": "k"}) == 1  # missing piece
    assert evaluate.board_errors(gt, {**gt, "h5": "q"}) == 1  # hallucinated piece
    assert evaluate.board_errors(gt, {**gt, "a2": "p"}) == 1  # wrong colour


def test_summarise_errors() -> None:
    s = evaluate.summarise_errors([0, 0, 1, 4])
    assert s["boards_0_errors_pct"] == 50.0 and s["boards_le1_errors_pct"] == 75.0
    assert s["mean_errors_per_board"] == 1.25 and s["square_accuracy_pct"] == pytest.approx(
        98.05, 0.01
    )


def test_corner_error_normalised_by_diagonal() -> None:
    gt = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    assert evaluate.corner_error(gt, gt) == 0
    assert evaluate.corner_error(gt + 10, gt) == pytest.approx(
        np.hypot(10, 10) / np.hypot(100, 100)
    )
    assert evaluate.corner_error(None, gt) is None


def test_foreshortening_ratio() -> None:
    sq = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    assert evaluate.foreshortening(sq) == 1.0
    trap = np.array([[25, 0], [75, 0], [100, 100], [0, 100]], float)
    assert evaluate.foreshortening(trap) == pytest.approx(0.5)


def _make_run(root, name, task, smoke=False):
    d = root / name
    d.mkdir(parents=True)
    (d / f"{name}.pt").write_bytes(b"weights")
    metrics = {
        "name": name,
        "task": task,
        "model": "m.pt",
        "imgsz": 640,
        "params": 2_600_000,
        "train_seconds": 600,
        "smoke": smoke,
        "weights": f"{name}.pt",
        "val": {"mAP50": 0.9, "mAP50-95": 0.7},
        "test": {"mAP50": 0.8, "mAP50-95": 0.6},
    }
    (d / "metrics.json").write_text(json.dumps(metrics))


def test_training_runs_collect_ignores_smoke(tmp_path) -> None:
    _make_run(tmp_path, "det_n_640", "detect")
    _make_run(tmp_path, "det_smoke", "detect", smoke=True)
    rows = training_runs.collect(tmp_path)
    assert [r["run"] for r in rows] == ["det_n_640"]
    assert rows[0]["test_mAP50-95"] == 0.6 and rows[0]["params_m"] == 2.6


def test_colab_bundle_contains_weights_and_manifest(tmp_path) -> None:
    import zipfile

    _make_run(tmp_path / "runs", "det_n_640", "detect")
    out = colab_runner.bundle(
        tmp_path / "runs", ["det_n_640", "missing"], tmp_path / "b.zip", False
    )
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert {"det_n_640/det_n_640.pt", "det_n_640/metrics.json", "manifest.json"} <= names
    assert manifest["kind"] == "trained" and "missing" not in manifest["runs"]
    assert manifest["runs"]["det_n_640"]["sha256"] == colab_runner.sha256(
        tmp_path / "runs/det_n_640/det_n_640.pt"
    )


def _board_image() -> np.ndarray:
    """Synthetic scene: wood-coloured table with a printed 8x8 board (white mat, margin)."""
    img = np.full((900, 1200, 3), (60, 110, 170), np.uint8)
    cv2.rectangle(img, (250, 100), (950, 800), (245, 245, 245), -1)  # mat
    x0, y0, cell = 300, 150, 75
    for r in range(8):
        for c in range(8):
            col = (30, 30, 30) if (r + c) % 2 else (240, 240, 240)
            cv2.rectangle(
                img,
                (x0 + c * cell, y0 + r * cell),
                (x0 + (c + 1) * cell, y0 + (r + 1) * cell),
                col,
                -1,
            )
    return img


def test_classic_corners_finds_clean_synthetic_board() -> None:
    res = corners_classic.find_corners(_board_image())
    assert res is not None
    gt = np.array([[300, 150], [900, 150], [900, 750], [300, 750]], float)
    assert np.mean(np.linalg.norm(res.corners - gt, axis=1)) < 25


def test_classic_corners_none_on_blank_image() -> None:
    assert corners_classic.find_corners(np.full((400, 400, 3), 128, np.uint8)) is None


def test_bootstrap_ci_brackets_point_estimate() -> None:
    from ml.stats import bootstrap_ci

    errs = [0] * 40 + [2] * 60
    games = [i % 4 for i in range(100)]
    ci = bootstrap_ci(errs, games, n=300)
    lo, hi = ci["board_bootstrap"]["boards_0_errors_pct"]
    assert lo <= 40 <= hi and ci["n_games"] == 4
    glo, ghi = ci["game_cluster_bootstrap"]["mean_errors_per_board"]
    assert glo <= 1.2 <= ghi
