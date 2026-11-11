"""Run all training jobs sequentially with resume, then bundle the best weights for download.

Used by ``notebooks/train_colab.ipynb`` (project dir lives on Google Drive so that a disconnected
session can resume), and runnable locally with ``--dry-run`` to validate the flow on a laptop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
import zipfile
from pathlib import Path

from ml import train

DEFAULT_RUNS = ["det_n_640", "corners_n_640", "det_n_960", "det_s_640"]  # priority order


def sha256(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def bundle(project: Path, runs: list[str], out_zip: Path, smoke: bool) -> Path:
    """Zip best weights + metrics.json of every finished run and a manifest."""
    manifest: dict = {"kind": "smoke" if smoke else "trained", "runs": {}}
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in runs:
            mpath = project / name / "metrics.json"
            if not mpath.exists():
                continue
            m = json.loads(mpath.read_text())
            pt = project / name / m["weights"]
            zf.write(pt, f"{name}/{pt.name}")
            zf.write(mpath, f"{name}/metrics.json")
            manifest["runs"][name] = {
                "task": m["task"],
                "imgsz": m["imgsz"],
                "weights": f"{name}/{pt.name}",
                "sha256": sha256(pt),
                "val": {k: v for k, v in m.get("val", {}).items() if k != "per_class"},
            }
        full = project / "full_test_eval.json"
        if full.exists():
            zf.write(full, "full_test_eval.json")
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    return out_zip


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, required=True, help="output dir (Drive on Colab)")
    ap.add_argument("--data-root", type=Path, default=Path("data/yolo"))
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--zip-out", type=Path, default=Path("chessvision_weights.zip"))
    ap.add_argument(
        "--dry-run", action="store_true", help="tiny smoke configs on local sample data"
    )
    ap.add_argument("--bundle-only", action="store_true", help="only (re)build the zip")
    a = ap.parse_args()

    runs = a.runs
    if a.dry_run:
        runs = ["smoke_det", "smoke_corners"]
    if a.bundle_only:
        names = [train.load_config(Path("configs") / f"{n}.yaml")["name"] for n in runs]
        print("bundle:", bundle(a.project, names, a.zip_out, smoke=a.dry_run))
        return
    failures: dict = {}
    run_names: list[str] = []
    for name in runs:
        cfg = train.load_config(Path("configs") / f"{name}.yaml")
        run_names.append(cfg["name"])
        data = a.data_root / ("detect.yaml" if cfg["task"] == "detect" else "pose.yaml")
        try:
            train.run(cfg, data, a.project, device="cpu" if a.dry_run else None, smoke=a.dry_run)
        except Exception:  # keep going: one failed run must not waste the other runs
            failures[name] = traceback.format_exc()
            print(f"!! run {name} failed:\n{failures[name]}")
    out = bundle(a.project, run_names, a.zip_out, smoke=a.dry_run)
    (a.project / "failures.json").write_text(json.dumps(failures, indent=2))
    print("bundle:", out, "| failed runs:", list(failures) or "none")
    if len(failures) == len(run_names):
        raise SystemExit("all runs failed")


if __name__ == "__main__":
    main()
