"""CPU benchmark: PyTorch vs ONNX FP32 vs ONNX INT8 (vs CoreML) - latency and accuracy.

    python -m ml.benchmark [--n 60] [--threads N]

Latency is measured per image for the whole pipeline (corner model + detector + geometry) with the
image already decoded, after warm-up. Accuracy is the end-to-end board accuracy on the *same*
images. Output: ``reports/benchmark.md|json|png`` (``reports/benchmark_smoke.*`` for smoke weights).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ml.chessvision.onnx_infer import OnnxCornerModel, OnnxDetector  # noqa: E402
from ml.chessvision.pipeline import BoardNotFoundError, Recognizer, read_manifest  # noqa: E402
from ml.chessvision.torch_infer import TorchCornerModel, TorchDetector  # noqa: E402
from ml.data.chessred import Annotations  # noqa: E402
from ml.evaluate import board_errors, gt_squares  # noqa: E402


def machine_spec() -> dict:
    """Describe the machine, so numbers can be interpreted."""
    import onnxruntime
    import torch

    def sysctl(k: str) -> str:
        try:
            return subprocess.check_output(["sysctl", "-n", k], text=True).strip()
        except Exception:  # noqa: BLE001
            return "n/a"

    mem = sysctl("hw.memsize")
    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "platform": platform.platform(),
        "cpu_cores_physical": sysctl("hw.physicalcpu"),
        "ram_gb": round(int(mem) / 2**30) if mem.isdigit() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
    }


def build_backends(wd: Path, manifest: dict, threads: int | None) -> dict[str, Recognizer]:
    """Create one Recognizer per available backend."""
    di, ci = manifest["imgsz"]["detector"], manifest["imgsz"]["corners"]
    det_pt, cor_pt = wd / f"{manifest['detector']}.pt", wd / f"{manifest['corners']}.pt"
    # corner_mode="learned_only": no classical fallback, so latency reflects the models alone
    lo = "learned_only"
    out: dict[str, Recognizer] = {
        "PyTorch FP32": Recognizer(TorchDetector(det_pt, di), TorchCornerModel(cor_pt, ci), lo),
        "ONNX FP32": Recognizer(
            OnnxDetector(wd / "detector.onnx", threads=threads),
            OnnxCornerModel(wd / "corners.onnx", threads=threads),
            lo,
        ),
        "ONNX INT8": Recognizer(
            OnnxDetector(wd / "detector_int8.onnx", threads=threads),
            OnnxCornerModel(wd / "corners_int8.onnx", threads=threads),
            lo,
        ),
    }
    if (wd / "detector.mlpackage").exists() and (wd / "corners.mlpackage").exists():
        try:
            out["CoreML FP16"] = Recognizer(
                TorchDetector(wd / "detector.mlpackage", di),
                TorchCornerModel(wd / "corners.mlpackage", ci),
                lo,
            )
        except Exception as e:  # noqa: BLE001
            print("CoreML backend unavailable:", e)
    return out


def bench_one(rec: Recognizer, imgs: list[np.ndarray], gts: list[dict], warmup: int = 3) -> dict:
    """Latency percentiles and accuracy for one backend."""
    # Importing Ultralytics disables OpenCV threading (cv2.setNumThreads(0)); the API image has no
    # Ultralytics, so re-enable it to measure the pipeline as deployed.
    cv2.setNumThreads(cv2.getNumberOfCPUs())
    for im in imgs[:warmup]:
        try:
            rec.recognize(im)
        except BoardNotFoundError:
            pass
    lat, errs = [], []
    for im, gt in zip(imgs, gts, strict=True):
        t0 = time.perf_counter()
        try:
            r = rec.recognize(im)
            pred = {s: v.symbol for s, v in r.reconstruction.squares.items()}
        except BoardNotFoundError:
            pred = {}
        lat.append((time.perf_counter() - t0) * 1000)
        errs.append(board_errors(gt, pred))
    e = np.array(errs)
    return {
        "p50_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "mean_ms": round(float(np.mean(lat)), 1),
        "images": len(imgs),
        "boards_0_errors_pct": round(100 * float((e == 0).mean()), 2),
        "mean_errors_per_board": round(float(e.mean()), 3),
        "square_accuracy_pct": round(100 * (1 - e.mean() / 64), 2),
    }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-dir", type=Path, default=Path("weights"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--threads", type=int, default=None, help="ORT intra-op threads")
    a = ap.parse_args()
    manifest = read_manifest(a.weights_dir)
    smoke = manifest.get("kind") != "trained"
    tag = "_smoke" if smoke else ""

    ann = Annotations(a.data_root / "raw" / "annotations.json")
    ids = json.loads((a.data_root / "local_manifest.json").read_text())["eval_test"]
    ids = ids[:: max(1, len(ids) // a.n)][: a.n]
    imgs = [cv2.imread(str(a.data_root / "raw/images" / ann.images[i]["file_name"])) for i in ids]
    gts = [gt_squares(ann, i) for i in ids]

    results = {}
    for name, rec in build_backends(a.weights_dir, manifest, a.threads).items():
        print("benchmarking", name)
        results[name] = bench_one(rec, imgs, gts)
    info = json.loads((a.weights_dir / "export_info.json").read_text())
    spec = machine_spec()
    out = Path("reports")
    payload = {
        "weights_kind": manifest.get("kind"),
        "machine": spec,
        "backends": results,
        "model_sizes_mb": info["models"],
        "threads": a.threads or "default",
    }
    (out / f"benchmark{tag}.json").write_text(json.dumps(payload, indent=2))

    lines = []
    if smoke:
        lines += [
            "> **SMOKE-TEST WEIGHTS - latency is representative of the architecture, accuracy "
            "numbers are meaningless and must not be quoted.**",
            "",
        ]
    lines += [
        "# CPU benchmark",
        "",
        f"- Machine: {spec['chip']} ({spec['cpu_cores_physical']} physical cores, "
        f"{spec['ram_gb']} GB), {spec['platform']}",
        f"- Python {spec['python']}, torch {spec['torch']}, onnxruntime {spec['onnxruntime']}",
        f"- {len(ids)} ChessReD2K test images, whole pipeline per image (corner model + detector"
        f" + geometry), image pre-decoded; ORT threads: {payload['threads']}",
        "",
        "| Backend | p50 ms | p95 ms | boards 0 errors % | mean errors / board |",
        "|---|---|---|---|---|",
    ]
    for n, r in results.items():
        lines.append(
            f"| {n} | {r['p50_ms']} | {r['p95_ms']} | {r['boards_0_errors_pct']} | "
            f"{r['mean_errors_per_board']} |"
        )
    lines += ["", "Model sizes (MB):", "", "| Model | ONNX FP32 | ONNX INT8 |", "|---|---|---|"]
    for k, v in info["models"].items():
        lines.append(f"| {k} (imgsz {v['imgsz']}) | {v['onnx_fp32_mb']} | {v['onnx_int8_mb']} |")
    lines += ["", f"![latency](benchmark{tag}.png)", ""]
    (out / f"benchmark{tag}.md").write_text("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = list(results)
    x = np.arange(len(names))
    ax.bar(x - 0.2, [results[n]["p50_ms"] for n in names], 0.4, label="p50")
    ax.bar(x + 0.2, [results[n]["p95_ms"] for n in names], 0.4, label="p95")
    ax.set_xticks(x, names)
    ax.set_ylabel("latency per image (ms)")
    ax.set_title("End-to-end CPU latency" + (" (smoke weights)" if smoke else ""))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"benchmark{tag}.png", dpi=120)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
