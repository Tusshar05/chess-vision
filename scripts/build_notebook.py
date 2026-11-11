"""Generate notebooks/train_colab.ipynb (kept in code so it is reviewable and testable)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/Tusshar05/chess-vision.git"
cells: list[dict] = []


def md(text: str) -> None:
    """Append a markdown cell."""
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)})


def code(text: str) -> None:
    """Append a code cell."""
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": text.strip().splitlines(True),
        }
    )


md("""
# ChessVision - training on a free Colab T4

This notebook trains everything the project needs and gives you **one zip to download**:

| Run | What | Rough time on T4 |
|---|---|---|
| `det_n_640` | YOLO11n piece detector @ 640 px | <= 45 min (capped) |
| `corners_n_640` | YOLO11n-pose board-corner model | <= 30 min (capped) |
| `det_n_960` | YOLO11n detector @ 960 px (resolution comparison) | <= 45 min (capped) |
| `det_s_640` | YOLO11s detector @ 640 px (model-size comparison) | <= 45 min (capped) |

**Before you start:** `Runtime -> Change runtime type -> T4 GPU`, then `Runtime -> Run all`.
Total ~2.5 h; each run has a wall-clock cap so it cannot overrun. Runs are listed in priority order:
if you only have time for two, `det_n_640` and `corners_n_640` give the full working system.

**If the session disconnects:** just re-run all cells. Checkpoints are written to Google Drive every
epoch, finished runs are skipped, and the interrupted run resumes from `last.pt`.

**What to download at the end:** `chessvision_weights.zip` (the last cell triggers the download; it
is also saved to `MyDrive/chessvision/`). Unzip it into `release_assets/` in your local repo.
""")

code(
    """
import os, subprocess, sys, json
from pathlib import Path

# Set CHESSVISION_DRY_RUN=1 to run this notebook's logic locally (no GPU, no Drive, no clone).
DRY_RUN = os.environ.get("CHESSVISION_DRY_RUN") == "1"
REPO_URL = "__REPO__"
print("dry run:", DRY_RUN)

def sh(cmd: str, cwd: str | None = None) -> None:
    '''Run a shell command, streaming output; raise on failure.'''
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)
""".replace("__REPO__", REPO)
)

md("""
## Step 1 - check the GPU
Training on CPU would take days. This cell fails loudly if no GPU is attached.
""")
code("""
import torch
if DRY_RUN:
    print("dry run: skipping GPU requirement")
else:
    assert torch.cuda.is_available(), "No GPU! Runtime -> Change runtime type -> T4 GPU"
    print("GPU:", torch.cuda.get_device_name(0))
    sh("nvidia-smi")
""")

md("""
## Step 2 - mount Google Drive
All checkpoints and results go to `MyDrive/chessvision/`, so they survive a disconnect.
(You will be asked to authorise Drive access.)
""")
code("""
if DRY_RUN:
    DRIVE_DIR = Path("runs_smoke_notebook")
else:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = Path("/content/drive/MyDrive/chessvision")
RUNS_DIR = DRIVE_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
print("outputs ->", RUNS_DIR)
""")

md("""
## Step 3 - get the code
Clones the repository (or pulls if it is already there) and `cd`s into it.
""")
code("""
if DRY_RUN:
    REPO_DIR = Path.cwd()
else:
    REPO_DIR = Path("/content/chess-vision")
    if REPO_DIR.exists():
        sh("git pull --ff-only", cwd=str(REPO_DIR))
    else:
        sh(f"git clone {REPO_URL} {REPO_DIR}")
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))
print("working dir:", Path.cwd())
""")

md("""
## Step 4 - install requirements
Colab already ships PyTorch; we only add Ultralytics (YOLO), python-chess, MLflow and a few helpers.
""")
code("""
if not DRY_RUN:
    sh("pip install -q ultralytics python-chess mlflow matplotlib pyyaml requests "
       "opencv-python-headless")
""")

md("""
## Step 5 - download and convert ChessReD2K
Uses the repo's own data scripts: downloads `annotations.json` (22 MB) and `chessred2k.zip`
(4.7 GB, MD5-verified; this is the subset that has piece boxes and board corners), checks that the
official train/val/test split shares no game between splits, resizes images to <= 1280 px and writes
YOLO-format detection + corner-pose datasets under `data/yolo/`. Takes ~10-15 min the first time.

Dataset: Masouris & van Gemert, *End-to-End Chess Recognition*, VISAPP 2024 (4TU.ResearchData,
DOI 10.4121/99b5c721-280b-450b-b058-b2900b69a90f.v2).
""")
code("""
if DRY_RUN:
    DATA_ROOT = Path("data/yolo_smoke")  # produced by `make data`
else:
    DATA_ROOT = Path("data/yolo")
    if not (DATA_ROOT / "detect.yaml").exists():
        sh(f"{sys.executable} -m ml.data.prepare --mode full --root data")
print("dataset yaml:", DATA_ROOT / "detect.yaml")
""")

md("""
## Step 6 - train all models
Runs `ml/colab_runner.py`, which trains the four configs in `configs/` one after another. Each run:
logs to MLflow, validates on the val and test splits at the end, and writes `metrics.json`.
This is the long cell (~2.5 h). Per-epoch checkpoints are saved to Drive, so re-running the notebook
after a disconnect resumes where it stopped.
""")
code("""
flags = "--dry-run" if DRY_RUN else ""
sh(f"{sys.executable} -m ml.colab_runner --project {RUNS_DIR} --data-root {DATA_ROOT} "
   f"--zip-out {DRIVE_DIR}/chessvision_weights.zip {flags}")
""")

md("""
## Step 7 - results at a glance
Reads every `metrics.json` and prints a comparison table (640 vs 960, n vs s, corner model).
""")
code("""
rows = []
for p in sorted(RUNS_DIR.glob("*/metrics.json")):
    m = json.loads(p.read_text())
    v = m.get("test") or m.get("val") or {}
    rows.append((m["name"], m["imgsz"], round(m["train_seconds"] / 60, 1),
                 round(v.get("mAP50", v.get("pose_mAP50", float("nan"))), 4),
                 round(v.get("mAP50-95", v.get("pose_mAP50-95", float("nan"))), 4)))
print(f"{'run':<16}{'imgsz':>6}{'min':>8}{'mAP50':>9}{'mAP50-95':>10}")
for r in rows:
    print(f"{r[0]:<16}{r[1]:>6}{r[2]:>8}{r[3]:>9}{r[4]:>10}")
""")

md("""
## Step 8 - download the weights
This triggers a browser download of **`chessvision_weights.zip`** (best weights of every finished
run, each run's `metrics.json`, and a `manifest.json` with SHA-256 checksums). A copy stays on
Drive at `MyDrive/chessvision/chessvision_weights.zip`.

Then, on your Mac: unzip into `release_assets/` and run
`make weights eval export bench report` (see `docs/RUN_ORDER.md`).
""")
code("""
zip_path = DRIVE_DIR / "chessvision_weights.zip"
print(zip_path, zip_path.stat().st_size // 1e6, "MB")
if not DRY_RUN:
    from google.colab import files
    files.download(str(zip_path))
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path("notebooks/train_colab.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote notebooks/train_colab.ipynb with", len(cells), "cells")
