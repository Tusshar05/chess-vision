"""Optional end-to-end evaluation on the FULL ChessReD test split (2,129 images, 20 games).

Meant for Colab (needs ~5 GB of test images, fetched image-by-image from the remote zip). The full
split has piece-to-square labels for every image but corner labels only for the ChessReD2K part, so
this reports the *auto-orientation* pipeline only and splits results into "2K subset" vs "rest".
Writes ``full_test_eval.json`` (picked up by ``make report`` from ``release_assets/``).

    python -m ml.evaluate_full --project <runs dir> --out <runs dir>/full_test_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from ml.chessvision.pipeline import BoardNotFoundError, Recognizer
from ml.data.chessred import CHESSRED_FULL, Annotations, download, fetch_from_remote_zip
from ml.evaluate import board_errors, gt_squares, summarise_errors
from ml.stats import bootstrap_ci


def select_models(project: Path) -> tuple[dict, dict]:
    """Best detector and corner run (by validation mAP50-95) from the runs' metrics.json files."""
    runs = [json.loads(p.read_text()) for p in project.glob("*/metrics.json")]
    runs = [m for m in runs if not m.get("smoke")] or runs
    det = max((m for m in runs if m["task"] == "detect"), key=lambda m: m["val"]["mAP50-95"])
    cor = max((m for m in runs if m["task"] == "pose"), key=lambda m: m["val"]["pose_mAP50-95"])
    return det, cor


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--local-only",
        action="store_true",
        help="dry run: use only images already in data/raw/images",
    )
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    from ml.chessvision.torch_infer import TorchCornerModel, TorchDetector

    ann = Annotations(
        download(__import__("ml.data.chessred", fromlist=["x"]).ANNOTATIONS, a.root / "raw")
    )
    det_m, cor_m = select_models(a.project)
    rec = Recognizer(
        TorchDetector(
            a.project / det_m["name"] / det_m["weights"], det_m["imgsz"], device=a.device
        ),
        TorchCornerModel(
            a.project / cor_m["name"] / cor_m["weights"], cor_m["imgsz"], device=a.device
        ),
    )
    ids = ann.splits_full["test"][: a.limit]
    img_dir = a.root / "raw" / "images_fulltest"
    if a.local_only:
        img_dir = a.root / "raw" / "images"
        ids = [i for i in ids if (img_dir / ann.images[i]["file_name"]).exists()]
    else:
        fetch_from_remote_zip(CHESSRED_FULL, [ann.images[i]["file_name"] for i in ids], img_dir)
    two_k = set(ann.splits_2k["test"])
    errs, games, in2k = [], [], []
    for iid in ids:
        img = cv2.imread(str(img_dir / ann.images[iid]["file_name"]))
        try:
            r = rec.recognize(img)
            pred = {s: v.symbol for s, v in r.reconstruction.squares.items()}
        except BoardNotFoundError:
            pred = {}
        errs.append(board_errors(gt_squares(ann, iid), pred))
        games.append(ann.game_of(iid))
        in2k.append(iid in two_k)
    sel = [e for e, k in zip(errs, in2k, strict=True) if k]
    rest = [e for e, k in zip(errs, in2k, strict=True) if not k]
    out = {
        "eval_set": "FULL ChessReD test split (official, 20 games)",
        "detector": det_m["name"],
        "corners": cor_m["name"],
        "smoke": bool(det_m.get("smoke")),
        "orientation": "auto (no corner labels outside ChessReD2K)",
        "full_test": {**summarise_errors(errs), "ci95": bootstrap_ci(errs, games)},
        "chessred2k_test_subset": summarise_errors(sel) if sel else None,
        "full_test_minus_2k": summarise_errors(rest) if rest else None,
    }
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
