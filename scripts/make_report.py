"""Regenerate the results blocks of README.md from files under reports/.

Every number comes from a JSON/CSV written by the evaluation scripts. Blocks that depend on
trained weights show "Pending" until ``reports/eval/WEIGHTS.json`` (and the benchmark JSON) say
``"kind": "trained"`` - smoke-test output can never reach the README.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

README = Path("README.md")
R = Path("reports")
PENDING = (
    "_Pending — run Colab training and `make weights eval export bench report` "
    "(see [docs/RUN_ORDER.md](docs/RUN_ORDER.md))._"
)


def load(p: Path) -> dict | list | None:
    """Parse JSON if the file exists."""
    return json.loads(p.read_text()) if p.exists() else None


def trained(p: Path) -> bool:
    """True if the JSON at ``p`` was produced with trained weights."""
    d = load(p)
    return bool(d) and d.get("kind", d.get("weights_kind")) == "trained"


def dataset_block() -> str:
    """Dataset statistics (computed from the real annotation file)."""
    s = load(R / "data/summary.json")
    if not s:
        return PENDING
    rows = ["| split | images | games | piece instances |", "|---|---|---|---|"]
    for k, v in s["splits"].items():
        rows.append(f"| {k} | {v['images']:,} | {v['games']} | {v['pieces']:,} |")
    k2 = s["chessred2k_images"]
    return (
        "\n".join(rows)
        + f"\n\nBoxes and corners exist only for ChessReD2K ({sum(k2.values()):,} images: "
        f"train {k2['train']} / val {k2['val']} / test {k2['test']}); detector and corner models "
        "train on it. Splits are by game (verified at every data-prep run).\n\n"
        "![class distribution](reports/data/class_distribution.png)"
    )


def geometry_block() -> str:
    """Model-free geometry ceiling (ground-truth boxes + corners)."""
    g = load(R / "eval/geometry_oracle.json")
    if not g:
        return PENDING
    rows = [
        "| anchor | boards, 0 errors | mean errors / board | auto-orientation |",
        "|---|---|---|---|",
    ]
    for k, v in g.items():
        rows.append(
            f"| {k} | {v['perfect_boards_pct']}% | {v['mean_errors_per_board']} | "
            f"{v['auto_orientation_accuracy_pct']}% |"
        )
    n = next(iter(g.values()))["images"]
    return "\n".join(rows) + f"\n\n{n} ChessReD2K test images, ground-truth boxes and corners."


def results_block() -> str:
    """Detector + end-to-end + corner comparison + paper comparison (trained weights only)."""
    if not trained(R / "eval/WEIGHTS.json"):
        return PENDING
    d, e, c = (
        load(R / f"eval/{n}.json") for n in ("detector_metrics", "e2e_summary", "corner_comparison")
    )
    w = load(R / "eval/WEIGHTS.json")
    out = [
        f"Weights: `{w['detector']}` + `{w['corners']}` (selected by validation mAP50-95).",
        "",
        "**Piece detector** (ChessReD2K test split)",
        "",
        "| mAP50 | mAP50-95 | precision | recall |",
        "|---|---|---|---|",
        f"| {d['mAP50']} | {d['mAP50-95']} | {d['precision']} | {d['recall']} |",
        "",
        "**End-to-end board accuracy**",
        "",
        "| setting | boards 0 errors | boards ≤1 error | mean errors / board | square acc.* |",
        "|---|---|---|---|---|",
    ]
    for k, label in (
        ("end_to_end_auto_orientation", "learned corners, auto orientation"),
        ("end_to_end_given_orientation", "learned corners, given orientation"),
        ("end_to_end_classic_corners_given_orientation", "OpenCV corners, given orientation"),
    ):
        s = e[k]
        out.append(
            f"| {label} | {s['boards_0_errors_pct']}% | {s['boards_le1_errors_pct']}% | "
            f"{s['mean_errors_per_board']} | {s['square_accuracy_pct']}% |"
        )
    out += [
        "",
        "\\*all 64 squares counted, empties included.",
        "",
        "**Board corners: learned vs classical OpenCV** (error as % of board diagonal)",
        "",
        "| method | found | median err | mean err | within 2% | within 5% |",
        "|---|---|---|---|---|---|",
    ]
    for k, label in (
        ("learned_with_fallback", "YOLO11n-pose (+fallback)"),
        ("classic_opencv", "OpenCV baseline"),
    ):
        s = c[k]
        out.append(
            f"| {label} | {s['found_pct']}% | {s['median_err_pct_of_diag']}% | "
            f"{s['mean_err_pct_of_diag']}% | {s['within_2pct_diag']}% | "
            f"{s['within_5pct_diag']}% |"
        )
    ref = e["paper_reference"]
    out += [
        "",
        f"**Comparison with the ChessReD paper:** the paper reports "
        f"{ref['board_accuracy_pct']}% of boards fully correct (Masouris & van Gemert, 2024). "
        f"Ours: {e['end_to_end_auto_orientation']['boards_0_errors_pct']}%. {ref['note']}",
        "",
        "![confusion matrix](reports/eval/confusion_matrix.png)",
        "",
        "See [reports/eval/error_analysis.md](reports/eval/error_analysis.md) and the "
        "failure gallery in `reports/eval/failure_gallery/`.",
    ]
    return "\n".join(out)


def training_block() -> str:
    """Training-run comparison from Colab metrics.json files."""
    p = R / "training/runs.md"
    return p.read_text() if p.exists() else PENDING


def bench_block() -> str:
    """CPU benchmark table (trained weights only)."""
    if not trained(R / "benchmark.json"):
        return PENDING
    b = load(R / "benchmark.json")
    m = b["machine"]
    rows = [
        f"{m['chip']}, {m['cpu_cores_physical']} cores, {m['ram_gb']} GB; whole pipeline per "
        "image (see [reports/benchmark.md](reports/benchmark.md)).",
        "",
        "| backend | p50 ms | p95 ms | boards 0 errors | mean errors / board |",
        "|---|---|---|---|---|",
    ]
    for k, v in b["backends"].items():
        rows.append(
            f"| {k} | {v['p50_ms']} | {v['p95_ms']} | {v['boards_0_errors_pct']}% | "
            f"{v['mean_errors_per_board']} |"
        )
    return "\n".join(rows) + "\n\n![latency](reports/benchmark.png)"


BLOCKS = {
    "DATASET": dataset_block,
    "GEOMETRY": geometry_block,
    "RESULTS": results_block,
    "TRAINING": training_block,
    "BENCH": bench_block,
}


def main() -> None:
    """Replace every ``<!-- NAME:START -->...<!-- NAME:END -->`` block in the README."""
    text = README.read_text()
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- {name}:START -->)(.*?)(<!-- {name}:END -->)", re.S)
        if not pat.search(text):
            print(f"warning: no {name} block in README")
            continue
        text = pat.sub(lambda m, f=fn: f"{m.group(1)}\n{f()}\n{m.group(3)}", text)
    README.write_text(text)
    print("README results blocks regenerated")


if __name__ == "__main__":
    main()
