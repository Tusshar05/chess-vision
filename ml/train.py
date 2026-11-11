"""Config-driven training for the piece detector and the board-corner pose model.

Usage::

    python -m ml.train --config configs/det_n_640.yaml --data data/yolo/detect.yaml --project runs

Each run writes ``<project>/<name>/metrics.json`` and copies the best weights to
``<project>/<name>/<name>.pt``. Runs resume automatically from ``last.pt`` if interrupted
(Colab disconnects) and are skipped if ``metrics.json`` already exists.
Training is logged to MLflow (local SQLite store ``mlflow.db`` unless MLFLOW_TRACKING_URI is set).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

TASK_DEFAULT_MODEL = {"detect": "yolo11n.pt", "pose": "yolo11n-pose.pt"}


def pick_device(requested: str | None = None) -> str:
    """cuda if available, else mps on Apple Silicon, else cpu (unless explicitly requested)."""
    if requested:
        return requested
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: Path) -> dict:
    """Load a YAML run config and fill defaults."""
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("task", "detect")
    cfg.setdefault("model", TASK_DEFAULT_MODEL[cfg["task"]])
    cfg.setdefault("train_args", {})
    return cfg


def _metrics_from(results, task: str) -> dict:
    """Extract headline numbers from an Ultralytics validation result."""
    if task == "detect":
        box = results.box
        names = results.names
        per_class = {
            names[int(c)]: {
                "precision": float(box.p[i]),
                "recall": float(box.r[i]),
                "ap50": float(box.ap50[i]),
                "ap50_95": float(box.ap[i]),
            }
            for i, c in enumerate(box.ap_class_index)
        }
        return {
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
            "per_class": per_class,
        }
    return {
        "box_mAP50": float(results.box.map50),
        "pose_mAP50": float(results.pose.map50),
        "pose_mAP50-95": float(results.pose.map),
    }


def run(
    cfg: dict, data: Path, project: Path, device: str | None = None, smoke: bool = False
) -> dict:
    """Train (or resume) one run and return its metrics dict."""
    from ultralytics import YOLO

    name = cfg["name"]
    run_dir = project / name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[{name}] already finished, skipping")
        return json.loads(metrics_path.read_text())

    device = pick_device(device or cfg.get("device"))
    args = dict(
        data=str(data),
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        batch=cfg.get("batch", 16),
        project=str(project.resolve()),
        name=name,
        exist_ok=True,
        device=device,
        workers=cfg.get("workers", 2),
        patience=cfg.get("patience", 30),
        seed=0,
        plots=not smoke,
        **cfg["train_args"],
    )
    if cfg.get("time_hours"):
        args["time"] = cfg["time_hours"]  # hard wall-clock budget per run (overrides epochs)

    last = run_dir / "weights" / "last.pt"
    t0 = time.time()
    if last.exists():
        print(f"[{name}] resuming from {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(cfg["model"])
        model.train(**args)
    train_s = time.time() - t0

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        best = last
    final = YOLO(str(best))
    metrics: dict = {
        "name": name,
        "task": cfg["task"],
        "model": cfg["model"],
        "imgsz": cfg["imgsz"],
        "epochs_configured": cfg["epochs"],
        "train_seconds": round(train_s, 1),
        "device": device,
        "smoke": smoke,
        "params": int(sum(p.numel() for p in final.model.parameters())),
    }
    ydata = yaml.safe_load(Path(data).read_text())
    for split in ("val", "test"):
        if split in ydata:
            res = final.val(
                data=str(data),
                split=split,
                imgsz=cfg["imgsz"],
                device=device,
                plots=False,
                verbose=False,
            )
            metrics[split] = _metrics_from(res, cfg["task"])
    out_pt = run_dir / f"{name}.pt"
    out_pt.write_bytes(best.read_bytes())
    metrics["weights"] = out_pt.name
    metrics_path.write_text(json.dumps(metrics, indent=2))
    _log_mlflow(cfg, metrics, run_dir, smoke)
    return metrics


def _log_mlflow(cfg: dict, metrics: dict, run_dir: Path, smoke: bool) -> None:
    """Record params/metrics in MLflow (best-effort: never fail a finished training run)."""
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("chessvision" + ("-smoke" if smoke else ""))
        with mlflow.start_run(run_name=cfg["name"]):
            mlflow.log_params({k: v for k, v in cfg.items() if not isinstance(v, dict)})
            flat = {}
            for split in ("val", "test"):
                for k, v in metrics.get(split, {}).items():
                    if isinstance(v, float):
                        flat[f"{split}_{k}"] = v
            mlflow.log_metrics(flat)
            mlflow.log_metric("train_seconds", metrics["train_seconds"])
            mlflow.log_artifact(str(run_dir / "metrics.json"))
    except Exception as e:  # noqa: BLE001
        print("mlflow logging skipped:", e)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="Ultralytics dataset yaml")
    ap.add_argument("--project", type=Path, default=Path("runs"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    m = run(load_config(a.config), a.data, a.project, a.device, a.smoke)
    print(json.dumps({k: v for k, v in m.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
