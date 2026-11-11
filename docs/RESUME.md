# Resume material

Fill the `[placeholders]` **only** with numbers from `reports/` after `make weights eval export bench
report`. Delete a bullet rather than inventing a number.

## Two-line description
ChessVision — photo-to-position chess recognition: YOLO11 piece + board-corner models, homography
square mapping, FEN + Stockfish analysis, served by FastAPI/Streamlit with CPU-optimised ONNX/INT8/CoreML inference.
Python · PyTorch/Ultralytics · ONNX Runtime · OpenCV · FastAPI · Docker/Kubernetes · GitHub Actions.

## Bullet templates (pick 3-4)
- Built an end-to-end chess recognition system (phone photo → FEN → Stockfish best move) from two
  YOLO11n models and a homography-based square-assignment stage, reaching **[X]% mAP50** on the
  piece detector and **[Y]% of boards with zero errors / [Z] mean errors per board** on a held-out,
  game-disjoint test set (ChessReD2K, [N] images).
- Eliminated train/test leakage by splitting **by game** instead of by image (consecutive photos of
  one game are near-duplicates), and verified it with an assertion on every data-prep run.
- Replaced a hand-tuned classical board finder with a learned corner model (YOLO11n-pose):
  median corner error **[A]% → [B]%** of the board diagonal and **[C]% → [D]%** boards-with-zero-errors.
- Optimised inference for CPU/edge: exported to ONNX FP32/INT8 and CoreML, cutting **p50 latency
  from [P] ms to [Q] ms** on an [Apple M-series] CPU with **[R] pp** change in board accuracy;
  serving image has no PyTorch dependency ([S] MB).
- Shipped it as a production-style service: FastAPI (Pydantic validation, Prometheus metrics,
  structured logs, SQLite history), Streamlit dashboard with editable FEN, Docker Compose,
  Kubernetes manifests with HPA, and GitHub Actions CI/CD (ruff, pytest [T]% coverage, GHCR images).
- Compared 640 vs 960 px and YOLO11n vs 11s on a free Colab T4 with resumable, MLflow-tracked
  runs; found [finding, e.g. 960 px adds [U] mAP50-95 for [V]× latency].

## Interview one-liner
"I built a system that turns a photo of a real chessboard into a FEN and a best move; the
interesting parts were the game-level data split, the homography geometry that turns boxes into
squares, and getting it to run fast on a CPU."
