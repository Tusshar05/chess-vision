"""Data preparation CLI.

``--mode local``  (Mac): annotations + small train/val sample (smoke tests) + the ChessReD2K test
                  split (evaluation), fetched image-by-image from the remote zip.
``--mode full``   (Colab): download the whole ChessReD2K archive, extract, and convert all splits.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from ml.data import report
from ml.data.chessred import (
    ANNOTATIONS,
    CHESSRED2K,
    Annotations,
    download,
    fetch_from_remote_zip,
    write_yolo_dataset,
)


def _spread(ids: list[int], n: int) -> list[int]:
    """n ids evenly spaced across the list (covers many games rather than the first one)."""
    step = max(1, len(ids) // n)
    return ids[::step][:n]


def run_local(root: Path, smoke_train: int, smoke_val: int, eval_n: int) -> None:
    """Prepare everything that fits on a laptop."""
    raw = root / "raw"
    ann = Annotations(download(ANNOTATIONS, raw))
    games = ann.assert_no_game_leakage()
    print("splits are disjoint by game:", {k: len(v) for k, v in games.items()}, "games")

    tr = _spread(ann.splits_2k["train"], smoke_train)
    va = _spread(ann.splits_2k["val"], smoke_val)
    te = _spread(ann.splits_2k["test"], eval_n)
    ids = tr + va + te
    imgs = root / "raw" / "images"
    fetch_from_remote_zip(CHESSRED2K, [ann.images[i]["file_name"] for i in ids], imgs)

    counts = write_yolo_dataset(
        ann, {"train": tr, "val": va}, imgs, root / "yolo_smoke"
    )
    counts |= write_yolo_dataset(ann, {"test": te}, imgs, root / "yolo_eval")
    (root / "local_manifest.json").write_text(
        json.dumps({"smoke_train": tr, "smoke_val": va, "eval_test": te, "counts": counts})
    )
    report.build(ann, imgs, Path("reports/data"), sample_ids=_spread(tr + va + te, 12))
    print("done:", counts)


def run_full(root: Path) -> None:
    """Colab path: whole ChessReD2K (4.7 GB) -> YOLO datasets for all three splits."""
    raw = root / "raw"
    ann = Annotations(download(ANNOTATIONS, raw))
    ann.assert_no_game_leakage()
    zpath = download(CHESSRED2K, raw)
    imgs = raw / "images"
    if not imgs.exists():
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    info.filename = Path(info.filename).name
                    zf.extract(info, imgs)
    counts = write_yolo_dataset(ann, ann.splits_2k, imgs, root / "yolo")
    print("done:", counts)
    report.build(ann, imgs, Path("reports/data"), sample_ids=_spread(ann.splits_2k["train"], 12))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["local", "full"], default="local")
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--smoke-train", type=int, default=40)
    ap.add_argument("--smoke-val", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=306, help="test images for local evaluation")
    a = ap.parse_args()
    if a.mode == "local":
        run_local(a.root, a.smoke_train, a.smoke_val, a.eval_n)
    else:
        run_full(a.root)


if __name__ == "__main__":
    main()
