"""Geometry-stage ceiling: ground-truth boxes + ground-truth corners through ``reconstruct``.

Isolates errors of the homography/square-assignment logic from model errors. Writes
``reports/eval/geometry_oracle.json``. Runs on the ChessReD2K test subset fetched by ``make data``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.chessvision.board import Detection, reconstruct
from ml.chessvision.constants import CLASS_TO_SYMBOL
from ml.data.chessred import CORNER_ORDER, Annotations


def run(root: Path = Path("data"), out: Path = Path("reports/eval/geometry_oracle.json")) -> dict:
    """Evaluate the geometry stage with perfect detections; returns and saves the metrics."""
    ann = Annotations(root / "raw" / "annotations.json")
    ids = json.loads((root / "local_manifest.json").read_text())["eval_test"]
    res = {}
    for name, lift in (
        ("bottom_centre (lift 0.00)", 0.0),
        ("lift 0.10", 0.10),
        ("lift 0.20", 0.20),
    ):
        errs, orient_ok = [], 0
        for iid in ids:
            c = [ann.corners[iid][k] for k in CORNER_ORDER]
            true_rot = min(range(4), key=lambda i: c[i][0] + c[i][1])
            dets = [
                Detection(
                    p["bbox"][0],
                    p["bbox"][1],
                    p["bbox"][0] + p["bbox"][2],
                    p["bbox"][1] + p["bbox"][3],
                    p["category_id"],
                    1.0,
                )
                for p in ann.pieces[iid]
                if "bbox" in p and p["category_id"] < 12
            ]
            rec = reconstruct(dets, np.array(c), rotation=true_rot, lift=lift)
            gt = {
                p["chessboard_position"]: CLASS_TO_SYMBOL[p["category_id"]]
                for p in ann.pieces[iid]
                if p["category_id"] < 12
            }
            pred = {k: v.symbol for k, v in rec.squares.items()}
            errs.append(sum(gt.get(q) != pred.get(q) for q in {*gt, *pred}))
            orient_ok += reconstruct(dets, np.array(c), lift=lift).rotation == true_rot
        e = np.array(errs)
        res[name] = {
            "images": len(ids),
            "perfect_boards_pct": round(100 * float((e == 0).mean()), 2),
            "mean_errors_per_board": round(float(e.mean()), 4),
            "auto_orientation_accuracy_pct": round(100 * orient_ok / len(ids), 2),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
