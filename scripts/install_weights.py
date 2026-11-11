"""Install Colab-trained weights from release_assets/ into weights/ and verify them.

Accepts either the unzipped layout (``release_assets/manifest.json`` + ``<run>/<run>.pt``) or
``release_assets/chessvision_weights.zip``. Verifies SHA-256 checksums, loads each model with
Ultralytics to check its task, picks the detector with the best *validation* mAP50-95 (test split is
never used for selection) and writes ``weights/manifest.json`` with ``kind: trained``.
If nothing is found, the smoke weights stay active.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ASSETS, WEIGHTS = Path("release_assets"), Path("weights")


def sha256(p: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    """Install trained weights; return a process exit code."""
    zips = sorted(ASSETS.glob("*.zip"))
    if not (ASSETS / "manifest.json").exists() and zips:
        print("extracting", zips[-1])
        zipfile.ZipFile(zips[-1]).extractall(ASSETS)
    mpath = ASSETS / "manifest.json"
    if not mpath.exists():
        print("No trained weights in release_assets/ - keeping smoke weights.")
        print(
            "Run notebooks/train_colab.ipynb on Colab, then unzip the result into release_assets/."
        )
        return 0
    manifest = json.loads(mpath.read_text())
    if manifest.get("kind") != "trained":
        print(f"release_assets/manifest.json is not a trained bundle (kind={manifest.get('kind')})")
        return 1
    from ultralytics import YOLO

    runs = manifest["runs"]
    for name, r in runs.items():
        pt = ASSETS / r["weights"]
        if sha256(pt) != r["sha256"]:
            print(f"CHECKSUM MISMATCH for {name}")
            return 1
        task = YOLO(str(pt)).task
        if task != r["task"]:
            print(f"{name}: expected task {r['task']}, model says {task}")
            return 1
        print(f"verified {name}: {task}, imgsz={r['imgsz']}")
    dets = {n: r for n, r in runs.items() if r["task"] == "detect"}
    cors = {n: r for n, r in runs.items() if r["task"] == "pose"}
    if not dets or not cors:
        print("need at least one detector and one corner run")
        return 1
    best_det = max(dets, key=lambda n: dets[n]["val"].get("mAP50-95", 0))
    best_cor = max(cors, key=lambda n: cors[n]["val"].get("pose_mAP50-95", 0))
    WEIGHTS.mkdir(exist_ok=True)
    for n in (best_det, best_cor):
        shutil.copy(ASSETS / runs[n]["weights"], WEIGHTS / f"{n}.pt")
    # stale ONNX/CoreML exports belong to the previous weights
    for stale in WEIGHTS.glob("*.onnx"):
        stale.unlink()
    for stale in WEIGHTS.glob("*.mlpackage"):
        shutil.rmtree(stale)
    (WEIGHTS / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "trained",
                "detector": best_det,
                "corners": best_cor,
                "imgsz": {"detector": dets[best_det]["imgsz"], "corners": cors[best_cor]["imgsz"]},
                "selected_by": "validation mAP50-95",
                "source_manifest_sha256": sha256(mpath),
            },
            indent=2,
        )
    )
    print(f"active: detector={best_det} corners={best_cor}. Next: make export eval bench report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
