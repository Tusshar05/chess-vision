"""Evaluation on the held-out ChessReD2K test images fetched by ``make data``.

Produces (under ``reports/eval/``, or ``reports/eval_smoke/`` for smoke weights):

* ``detector_metrics.json`` + ``confusion_matrix.{csv,png}``: mAP50, mAP50-95, per-class P/R.
* ``corner_comparison.json``: learned corner model vs the classical OpenCV baseline.
* ``e2e_summary.json`` / ``e2e_per_image.csv``: square-level accuracy, % boards with 0 / <=1
  errors, mean errors per board (auto orientation and given orientation), by camera.
* ``error_analysis.md`` and ``failure_gallery/``: where and why it fails.

Smoke-test weights are evaluated only to prove the code works; their numbers go to
``reports/eval_smoke/`` and are never used in the README.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ml.chessvision import corners_classic, render  # noqa: E402
from ml.chessvision.constants import CLASS_NAMES  # noqa: E402
from ml.chessvision.pipeline import BoardNotFoundError, Recognizer, read_manifest  # noqa: E402
from ml.data.chessred import CORNER_ORDER, Annotations, canonical_corners  # noqa: E402

PAPER_BOARD_ACC = 15.26  # % boards fully correct, Masouris & van Gemert 2024 (full test split)


def gt_squares(ann: Annotations, iid: int) -> dict[str, str]:
    """Ground-truth piece symbols by square (empty squares omitted)."""
    from ml.chessvision.constants import CLASS_TO_SYMBOL

    return {
        p["chessboard_position"]: CLASS_TO_SYMBOL[p["category_id"]]
        for p in ann.pieces[iid]
        if p["category_id"] < 12
    }


def board_errors(gt: dict[str, str], pred: dict[str, str]) -> int:
    """Number of the 64 squares whose content differs (piece type/colour or occupancy)."""
    return sum(gt.get(sq) != pred.get(sq) for sq in {*gt, *pred})


def detector_metrics(pt: Path, imgsz: int, data_yaml: Path, out: Path) -> dict:
    """Ultralytics validation on the test split + confusion matrix."""
    from ultralytics import YOLO

    res = YOLO(str(pt)).val(
        data=str(data_yaml), split="test", imgsz=imgsz, device="cpu", plots=False, verbose=False
    )
    box = res.box
    per = {
        res.names[int(c)]: {
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
            "ap50": round(float(box.ap50[i]), 4),
        }
        for i, c in enumerate(box.ap_class_index)
    }
    m = {
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per,
        "images": len(list((data_yaml.parent / "detect/images/test").glob("*.jpg"))),
    }
    out.joinpath("detector_metrics.json").write_text(json.dumps(m, indent=2))
    cm = np.asarray(res.confusion_matrix.matrix)
    labels = [*CLASS_NAMES, "background"]
    with open(out / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pred\\true", *labels])
        for i, row in enumerate(cm):
            w.writerow([labels[i], *row.astype(int)])
    norm = cm / np.maximum(cm.sum(0, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(norm, cmap="Blues")
    ax.set_xticks(range(13), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(13), labels, fontsize=8)
    ax.set_xlabel("true class")
    ax.set_ylabel("predicted class")
    ax.set_title("Detector confusion matrix (column-normalised, test split)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    return m


def corner_error(pred: np.ndarray | None, gt: np.ndarray) -> float | None:
    """Mean corner distance as a fraction of the ground-truth board diagonal."""
    if pred is None:
        return None
    diag = float(np.linalg.norm(gt[0] - gt[2]))
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)) / diag)


def summarise_errors(errs: list[int]) -> dict:
    """Headline board-level statistics from per-board error counts."""
    e = np.array(errs)
    return {
        "boards": len(e),
        "square_accuracy_pct": round(100 * (1 - e.mean() / 64), 2),
        "boards_0_errors_pct": round(100 * float((e == 0).mean()), 2),
        "boards_le1_errors_pct": round(100 * float((e <= 1).mean()), 2),
        "mean_errors_per_board": round(float(e.mean()), 3),
        "median_errors_per_board": float(np.median(e)),
    }


def foreshortening(gt_corners: np.ndarray) -> float:
    """Shorter/longer horizontal board edge: 1 = fronto-parallel, small = steep angle."""
    a = np.linalg.norm(gt_corners[1] - gt_corners[0])
    b = np.linalg.norm(gt_corners[2] - gt_corners[3])
    return float(min(a, b) / max(a, b))


def run_e2e(rec: Recognizer, ann: Annotations, ids: list[int], img_dir: Path, out: Path) -> dict:
    """End-to-end evaluation of ``rec`` on ``ids``; writes CSV, gallery and error analysis."""
    rows, cc, c_learned, c_classic = [], Counter(), [], []
    classic_rec = Recognizer(rec.detector, None, "classic")
    errs_auto, errs_given, errs_classic = [], [], []
    keep = []
    for iid in ids:
        img = cv2.imread(str(img_dir / ann.images[iid]["file_name"]))
        gtc = np.array(canonical_corners([ann.corners[iid][k] for k in CORNER_ORDER]), np.float32)
        true_rot = min(
            range(4),
            key=lambda i: (
                [ann.corners[iid][k] for k in CORNER_ORDER][i][0]
                + [ann.corners[iid][k] for k in CORNER_ORDER][i][1]
            ),
        )
        gt = gt_squares(ann, iid)
        try:
            r_auto = rec.recognize(img)
            pred = {s: v.symbol for s, v in r_auto.reconstruction.squares.items()}
            e_auto = board_errors(gt, pred)
            e_given = board_errors(
                gt,
                {
                    s: v.symbol
                    for s, v in rec.recognize(img, rotation=true_rot).reconstruction.squares.items()
                },
            )
            ce = corner_error(r_auto.corners, gtc)
            src = r_auto.corner_source
            rot_ok = r_auto.rotation == true_rot
        except BoardNotFoundError:
            e_auto = e_given = board_errors(gt, {})
            ce, src, rot_ok, r_auto = None, "none", False, None
        try:
            e_classic = board_errors(
                gt,
                {
                    s: v.symbol
                    for s, v in classic_rec.recognize(
                        img, rotation=true_rot
                    ).reconstruction.squares.items()
                },
            )
            cl_res = corners_classic.find_corners(img)
            c_classic.append(corner_error(cl_res.corners if cl_res else None, gtc))
        except BoardNotFoundError:
            e_classic = board_errors(gt, {})
            c_classic.append(None)
        c_learned.append(ce)
        errs_auto.append(e_auto)
        errs_given.append(e_given)
        errs_classic.append(e_classic)
        im = ann.images[iid]
        cc[im["camera"]] += 1
        rows.append(
            {
                "image_id": iid,
                "file": im["file_name"],
                "camera": im["camera"],
                "pieces": len(gt),
                "foreshortening": round(foreshortening(gtc), 3),
                "errors_auto": e_auto,
                "errors_given_orientation": e_given,
                "errors_classic_corners": e_classic,
                "corner_source": src,
                "corner_err_diag_frac": None if ce is None else round(ce, 4),
                "orientation_correct": rot_ok,
            }
        )
        keep.append((e_auto, iid, r_auto, gt))
    with open(out / "e2e_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def corner_stats(vals: list[float | None]) -> dict:
        ok = np.array([v for v in vals if v is not None])
        return {
            "found_pct": round(100 * len(ok) / len(vals), 2),
            "mean_err_pct_of_diag": round(100 * float(ok.mean()), 2) if len(ok) else None,
            "median_err_pct_of_diag": round(100 * float(np.median(ok)), 2) if len(ok) else None,
            "within_2pct_diag": round(100 * float((ok < 0.02).sum()) / len(vals), 2),
            "within_5pct_diag": round(100 * float((ok < 0.05).sum()) / len(vals), 2),
        }

    summary = {
        "images": len(ids),
        "end_to_end_auto_orientation": summarise_errors(errs_auto),
        "end_to_end_given_orientation": summarise_errors(errs_given),
        "end_to_end_classic_corners_given_orientation": summarise_errors(errs_classic),
        "orientation_auto_accuracy_pct": round(
            100 * np.mean([r["orientation_correct"] for r in rows]), 2
        ),
        "paper_reference": {
            "board_accuracy_pct": PAPER_BOARD_ACC,
            "note": "Masouris & van Gemert (VISAPP 2024), full 2,129-image test split, different "
            "model (end-to-end CNN) and no corner/orientation annotations used at test time; "
            "ours is the 306-image ChessReD2K test subset with a detector+geometry pipeline. "
            "Indicative only, not a controlled comparison.",
        },
    }
    out.joinpath("e2e_summary.json").write_text(json.dumps(summary, indent=2))
    out.joinpath("corner_comparison.json").write_text(
        json.dumps(
            {
                "learned_with_fallback": corner_stats(c_learned),
                "classic_opencv": corner_stats(c_classic),
            },
            indent=2,
        )
    )
    _error_analysis(rows, ann, keep, img_dir, out)
    return summary


def _bin_table(rows: list[dict], key, labels: list[str], title: str) -> str:
    """Markdown table of board stats per bin."""
    lines = [
        f"### {title}",
        "",
        "| group | boards | mean errors | % perfect |",
        "|---|---|---|---|",
    ]
    for lab in labels:
        sel = [r["errors_auto"] for r in rows if key(r) == lab]
        if sel:
            e = np.array(sel)
            lines.append(f"| {lab} | {len(e)} | {e.mean():.2f} | {100 * (e == 0).mean():.1f} |")
    return "\n".join(lines) + "\n"


def _error_analysis(rows, ann, keep, img_dir: Path, out: Path) -> None:
    """Error breakdown tables + failure gallery of the 12 worst boards."""
    fs = sorted(r["foreshortening"] for r in rows)
    lo, hi = fs[len(fs) // 3], fs[2 * len(fs) // 3]

    def ang(r: dict) -> str:
        if r["foreshortening"] <= lo:
            return "steep (low foreshortening ratio)"
        return "moderate" if r["foreshortening"] <= hi else "shallow (near fronto-parallel)"

    def occ(r: dict) -> str:
        return (
            "<=8 pieces"
            if r["pieces"] <= 8
            else ("9-20 pieces" if r["pieces"] <= 20 else ">20 pieces")
        )

    md = [
        "# Error analysis",
        "",
        "Auto-generated by `ml/evaluate.py` from `e2e_per_image.csv`. "
        "Errors = wrong squares per board (auto orientation).",
        "",
    ]
    md.append(
        _bin_table(rows, lambda r: r["camera"], sorted({r["camera"] for r in rows}), "By camera")
    )
    md.append(
        _bin_table(
            rows,
            ang,
            ["steep (low foreshortening ratio)", "moderate", "shallow (near fronto-parallel)"],
            "By viewing angle (proxy)",
        )
    )
    md.append(
        _bin_table(
            rows,
            occ,
            ["<=8 pieces", "9-20 pieces", ">20 pieces"],
            "By piece count (proxy for occlusion density)",
        )
    )
    md.append(
        _bin_table(
            rows, lambda r: r["corner_source"], ["learned", "classic", "none"], "By corner source"
        )
    )
    md.append(
        _bin_table(
            rows,
            lambda r: "correct" if r["orientation_correct"] else "wrong",
            ["correct", "wrong"],
            "By auto-orientation outcome",
        )
    )
    gdir = out / "failure_gallery"
    gdir.mkdir(exist_ok=True)
    conf_pairs: Counter = Counter()
    worst = sorted(keep, key=lambda k: -k[0])[:12]
    for _e, _iid, res, gt in keep:
        if res is None:
            continue
        pred = {s: v.symbol for s, v in res.reconstruction.squares.items()}
        for sq in {*gt, *pred}:
            if gt.get(sq) != pred.get(sq):
                conf_pairs[(gt.get(sq, "empty"), pred.get(sq, "empty"))] += 1
    md += [
        "### Most common square errors (truth -> predicted)",
        "",
        "| truth | predicted | count |",
        "|---|---|---|",
    ]
    md += [f"| {a} | {b} | {n} |" for (a, b), n in conf_pairs.most_common(12)]
    md += [
        "",
        "### Failure gallery",
        "",
        "Worst boards (annotated photo; see `failure_gallery/`):",
        "",
    ]
    for e, iid, res, _gt in worst:
        if res is None:
            continue
        img = cv2.imread(str(img_dir / ann.images[iid]["file_name"]))
        vis = render.annotate(img, res.corners, res.detections)
        vis = cv2.resize(vis, (800, 800))
        cv2.putText(
            vis,
            f"{ann.images[iid]['file_name']}  errors={e}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
        name = f"{e:02d}err_{Path(ann.images[iid]['file_name']).stem}.jpg"
        cv2.imwrite(str(gdir / name), vis)
        md.append(f"- `{name}`")
    md += [
        "",
        "## Analysis (fill in after reviewing the gallery with real weights)",
        "",
        "- **Steep angles:** _observations_",
        "- **Occlusion (hand/other pieces):** _observations_",
        "- **Lighting / glare:** _observations_",
        "- **Look-alike classes (bishop/pawn, queen/king):** _observations_",
        "- **Fixes tried / next:** _observations_",
        "",
    ]
    (out / "error_analysis.md").write_text("\n".join(md))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-dir", type=Path, default=Path("weights"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--precision", default="fp32", choices=["fp32", "int8"])
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    a = ap.parse_args()

    from ml.chessvision.pipeline import load_recognizer

    manifest = read_manifest(a.weights_dir)
    smoke = manifest.get("kind") != "trained"
    out = Path("reports/eval_smoke" if smoke else "reports/eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / "WEIGHTS.json").write_text(
        json.dumps(
            {
                "kind": manifest.get("kind"),
                "note": manifest.get("note", ""),
                "detector": manifest.get("detector"),
                "corners": manifest.get("corners"),
            },
            indent=2,
        )
    )
    ann = Annotations(a.data_root / "raw" / "annotations.json")
    ids = json.loads((a.data_root / "local_manifest.json").read_text())["eval_test"][: a.limit]
    imgsz = manifest.get("imgsz", {}).get("detector", 640)
    detector_metrics(
        a.weights_dir / f"{manifest['detector']}.pt",
        imgsz,
        a.data_root / "yolo_eval" / "detect.yaml",
        out,
    )
    rec = load_recognizer(a.weights_dir, precision=a.precision)
    s = run_e2e(rec, ann, ids, a.data_root / "raw" / "images", out)
    print(json.dumps(s, indent=2))
    if smoke:
        print("\n*** SMOKE WEIGHTS: numbers above are NOT results (written to", out, ")")


if __name__ == "__main__":
    main()
