"""Tiny smoke training of both models. Proves the pipeline end to end - NOT for reporting.

Outputs ``weights/detector_smoke.pt``, ``weights/corners_smoke.pt`` (+ ONNX via ``ml.export``) and
a ``weights/manifest.json`` marking them as ``kind: smoke``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ml import train

SMOKE_NOTE = "smoke-test, not for reporting"


def main(data_root: Path = Path("data/yolo_smoke"), project: Path = Path("runs_smoke")) -> None:
    """Run both smoke trainings on CPU and stage the weights."""
    weights = Path("weights")
    weights.mkdir(exist_ok=True)
    for cfg_name, data_yaml, out_name in (
        ("smoke_det", "detect.yaml", "detector_smoke.pt"),
        ("smoke_corners", "pose.yaml", "corners_smoke.pt"),
    ):
        cfg = train.load_config(Path("configs") / f"{cfg_name}.yaml")
        m = train.run(cfg, data_root / data_yaml, project, device="cpu", smoke=True)
        shutil.copy(project / cfg["name"] / m["weights"], weights / out_name)
    (weights / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "smoke",
                "note": SMOKE_NOTE,
                "detector": "detector_smoke",
                "corners": "corners_smoke",
                "imgsz": {"detector": 320, "corners": 320},
            },
            indent=2,
        )
    )
    print("smoke weights staged in", weights, f"({SMOKE_NOTE})")


if __name__ == "__main__":
    main()
