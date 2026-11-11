"""Collect Colab ``metrics.json`` files (640 vs 960, n vs s, corner model) into one table.

Reads ``release_assets/**/metrics.json`` and writes ``reports/training/runs.{json,md}``.
Smoke runs (``"smoke": true``) are ignored.
"""

from __future__ import annotations

import json
from pathlib import Path


def collect(assets: Path = Path("release_assets")) -> list[dict]:
    """Return one summary row per finished real training run."""
    rows = []
    for p in sorted(assets.glob("**/metrics.json")):
        m = json.loads(p.read_text())
        if m.get("smoke"):
            continue
        v, t = m.get("val", {}), m.get("test", {})
        rows.append(
            {
                "run": m["name"],
                "task": m["task"],
                "model": m["model"],
                "imgsz": m["imgsz"],
                "params_m": round(m["params"] / 1e6, 2),
                "train_min": round(m["train_seconds"] / 60, 1),
                "val_mAP50": round(v.get("mAP50", v.get("pose_mAP50", float("nan"))), 4),
                "val_mAP50-95": round(v.get("mAP50-95", v.get("pose_mAP50-95", float("nan"))), 4),
                "test_mAP50": round(t.get("mAP50", t.get("pose_mAP50", float("nan"))), 4),
                "test_mAP50-95": round(t.get("mAP50-95", t.get("pose_mAP50-95", float("nan"))), 4),
            }
        )
    return rows


def main(out: Path = Path("reports/training")) -> list[dict]:
    """Write the comparison files; no-op (returns []) if no real runs exist."""
    rows = collect()
    if not rows:
        print("no real training runs found in release_assets/ (smoke runs are ignored)")
        return rows
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps(rows, indent=2))
    head = (
        "| run | task | model | imgsz | params (M) | train min | val mAP50 | val mAP50-95 "
        "| test mAP50 | test mAP50-95 |"
    )
    md = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['run']} | {r['task']} | {r['model']} | {r['imgsz']} | {r['params_m']} | "
            f"{r['train_min']} | {r['val_mAP50']} | {r['val_mAP50-95']} | {r['test_mAP50']} | "
            f"{r['test_mAP50-95']} |"
        )
    (out / "runs.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return rows


if __name__ == "__main__":
    main()
