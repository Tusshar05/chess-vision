"""Assemble ``demo_build/`` - a self-contained folder to push to a Hugging Face Space.

Refuses smoke weights unless ``--allow-smoke`` (we never deploy smoke models publicly).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SPACE_README = """---
title: ChessVision
emoji: ♟️
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: agpl-3.0
---

# ChessVision demo

Photo of a physical chessboard -> FEN -> Stockfish best move. Code: https://github.com/Tusshar05/chess-vision

Trained on ChessReD (CC BY-NC-SA 4.0): Masouris & van Gemert, End-to-End Chess
Recognition, VISAPP 2024.
Ultralytics YOLO is AGPL-3.0; Stockfish (GPL-3.0) is run as a separate binary.
"""


def main() -> int:
    """Build the bundle; return exit code."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("demo_build"))
    ap.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--allow-smoke", action="store_true")
    a = ap.parse_args()
    manifest = json.loads(Path("weights/manifest.json").read_text())
    if manifest.get("kind") != "trained" and not a.allow_smoke:
        print(
            "Refusing: weights are smoke-test weights. "
            "Train on Colab, then run `make weights export`."
        )
        return 1
    suffix = "_int8" if a.precision == "int8" else ""
    out = a.out
    if out.exists():
        shutil.rmtree(out)
    (out / "models").mkdir(parents=True)
    for k in ("detector", "corners"):
        shutil.copy(f"weights/{k}{suffix}.onnx", out / "models" / f"{k}{suffix}.onnx")
    shutil.copy("weights/manifest.json", out / "models/manifest.json")
    shutil.copytree(
        "ml/chessvision",
        out / "ml/chessvision",
        ignore=shutil.ignore_patterns("__pycache__", "torch_infer.py"),
    )
    shutil.copy("ml/__init__.py", out / "ml/__init__.py")
    shutil.copytree("samples", out / "samples", ignore=shutil.ignore_patterns("*.json"))
    for f in ("app.py", "requirements.txt", "packages.txt"):
        shutil.copy(f"demo/{f}", out / f)
    (out / "README.md").write_text(SPACE_README)
    print("built", out, "- now push this folder to your Hugging Face Space (docs/DEPLOY_DEMO.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
