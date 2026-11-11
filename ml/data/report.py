"""Dataset report: class distribution, images per camera, annotated sample grid."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ml.data.chessred import CLASS_NAMES, CORNER_ORDER, Annotations  # noqa: E402


def build(ann: Annotations, image_dir: Path, out: Path, sample_ids: list[int]) -> None:
    """Write plots and summary.json under ``out`` (computed from the annotation file)."""
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"n_images": len(ann.images), "splits": {}}
    cls_by_split: dict[str, Counter] = {}
    cam_by_split: dict[str, Counter] = {}
    for split, ids in ann.splits_full.items():
        c, cam = Counter(), Counter()
        for i in ids:
            cam[ann.images[i]["camera"]] += 1
            for p in ann.pieces[i]:
                if p["category_id"] < 12:
                    c[CLASS_NAMES[p["category_id"]]] += 1
        cls_by_split[split], cam_by_split[split] = c, cam
        summary["splits"][split] = {
            "images": len(ids),
            "games": len({ann.game_of(i) for i in ids}),
            "pieces": sum(c.values()),
            "cameras": dict(cam),
        }
    summary["chessred2k_images"] = {k: len(v) for k, v in ann.splits_2k.items()}
    summary["images_with_bbox_or_corners"] = len(ann.corners)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.27
    for k, (split, c) in enumerate(cls_by_split.items()):
        ax.bar([x + k * w for x in range(12)], [c[n] for n in CLASS_NAMES], w, label=split)
    ax.set_xticks([x + w for x in range(12)], CLASS_NAMES, rotation=40, ha="right")
    ax.set_ylabel("piece instances (full ChessReD)")
    ax.set_title("Class distribution per official split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=130)
    plt.close(fig)

    cams = sorted({c for s in cam_by_split.values() for c in s})
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, (split, cam) in enumerate(cam_by_split.items()):
        ax.bar([x + k * w for x in range(len(cams))], [cam[c] for c in cams], w, label=split)
    ax.set_xticks([x + w for x in range(len(cams))], cams)
    ax.set_ylabel("images")
    ax.set_title("Images per camera")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "images_per_camera.png", dpi=130)
    plt.close(fig)

    shown = [i for i in sample_ids if (image_dir / ann.images[i]["file_name"]).exists()][:12]
    if shown:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for a in axes.ravel():
            a.axis("off")
        for a, iid in zip(axes.ravel(), shown, strict=False):
            img = cv2.cvtColor(cv2.imread(str(image_dir / ann.images[iid]["file_name"])),
                               cv2.COLOR_BGR2RGB)
            s = img.shape[1] / ann.images[iid]["width"]
            for p in ann.pieces[iid]:
                if "bbox" in p and p["category_id"] < 12:
                    x, y, bw, bh = (v * s for v in p["bbox"])
                    col = (255, 60, 60) if p["category_id"] >= 6 else (60, 220, 60)
                    cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), col, 3)
            if iid in ann.corners:
                pts = [tuple(int(v * s) for v in ann.corners[iid][k]) for k in CORNER_ORDER]
                cv2.polylines(img, [__import__("numpy").array(pts)], True, (255, 255, 0), 4)
            a.imshow(img)
            a.set_title(ann.images[iid]["file_name"], fontsize=9)
        fig.suptitle("ChessReD2K samples (green=white pieces, red=black, yellow=board corners)")
        fig.tight_layout()
        fig.savefig(out / "sample_grid.jpg", dpi=70)
        plt.close(fig)
