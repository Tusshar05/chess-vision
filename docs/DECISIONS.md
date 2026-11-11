# Decisions log

Autonomous decisions made while building ChessVision, with reasons. Newest at the bottom.

## Data
- **License discrepancy.** The task brief says ChessReD is CC BY 4.0. The 4TU.ResearchData API
  (queried 2026-11-11) lists **CC BY-NC-SA 4.0**. I documented the license as listed by the
  source. Consequences: attribution required, non-commercial use only, derivatives (e.g. sample
  images, anything derived from the data) must be shared alike. Fine for a portfolio project;
  do not use the dataset or models trained on it commercially without checking with the authors.
- **Dataset sizes** (4TU API): `chessred.zip` 24.6 GB, `chessred2k.zip` 4.68 GB,
  `annotations.json` 22 MB. Mac has ~12 GB free, so the Mac never downloads the full archive.
- **Only ChessReD2K has boxes and corners** (2,078 images, 20 games, official train/val/test =
  1442/330/306). So detector and corner training use ChessReD2K; piece-to-square ground truth
  (for FEN) exists for all 10,800 images.
- **Official split is by game** (train 60 / val 20 / test 20 games). `assert_no_game_leakage`
  verifies this on every run; ChessReD2K splits are sub-splits of the same games.
- **Remote range-reads instead of full downloads.** The server supports HTTP Range, so
  `fetch_from_remote_zip` reads the zip central directory and pulls individual JPEGs. The Mac
  fetches 40 train + 10 val images (smoke) and the 306-image ChessReD2K test split (eval).
  Colab downloads the 4.68 GB archive with the same module.
- **Images resized to max side 1280** at conversion (originals are 3072x3072 / ~3 MB). Labels are
  normalised, so unaffected; saves disk and training time. Supports imgsz 640 and 960 runs.
- **Corner labels are board-relative, not image-relative.** I verified: `top_left` is the a8
  corner in every image where a8 is occupied (834/834 checked), but only 522/2078 have it at the
  image's top-left, i.e. photos are taken from all four sides. Decision: train the corner model on
  an *image-space canonical order* (TL,TR,BR,BL starting from the corner with min x+y), and resolve
  which board corner is a8 afterwards from piece colours (4 rotations scored; user flag overrides).
  This keeps the keypoint task easy and makes `fliplr` augmentation valid (`flip_idx=[1,0,3,2]`).
- **Evaluation set**: ChessReD2K test split (306 images from **only 3 games**: ChessReD2K's test split is 3 of the 20 test games,
  so per-board results have wide error bars and one board/lighting setup can dominate), not the paper's
  full 2,129-image test split. Comparison with the paper is therefore indicative
  only; the README says so.

## Board logic
- **Anchor lift 0.10.** Piece anchor = bottom-centre of the box, lifted by 10% of box height
  (the base ellipse is slightly above the box's bottom edge). With *ground-truth* boxes and corners
  on the 306 test images this gives 100% perfect boards vs 98.7% for the plain bottom-centre
  (`reports/eval/geometry_oracle.json`). Caveat: chosen among 3 values on the same images it is
  reported on; it is a geometry constant, not a learned parameter, but treat the oracle as a ceiling.
- **Auto-orientation** scores the 4 rotations by "white pieces low, black high" (kings and pawns
  weighted double). Oracle accuracy 96.4% — it fails on sparse endgames (6-7 pieces). Mitigation:
  explicit `rotation` parameter in the API, and a rotate control in the dashboard.
- Castling rights are inferred only when king and rook sit on home squares; side to move defaults
  to White (a photo cannot tell) and is a request parameter.

## Inference, evaluation, benchmark
- **One downscale, then models.** Photos (3072 px) are area-downscaled once to a 1280 px long
  side — exactly how training images were prepared — and detections/corners are mapped back to the
  original coordinates. Avoids two expensive resizes and keeps train/inference preprocessing aligned.
- **ONNX pipeline, no PyTorch at serving time.** Ultralytics exports raw head tensors; decoding,
  per-class NMS (OpenCV) and pose decoding are implemented in `ml/chessvision/onnx_infer.py`.
- **INT8 = static QDQ, per-channel, calibrated on 50 train/val images** (never the test split).
- **Benchmark honesty.** Importing Ultralytics silently disables OpenCV threading, which made the
  resize look 7x slower in my first benchmark run; `ml/benchmark.py` re-enables it so the numbers
  reflect the deployed (Ultralytics-free) API. Benchmarks use `learned_only` corner mode so a
  silent fallback to the slow classical detector cannot pollute latency.
- **coremltools 9 needs numpy < 2.3** (export crashed on 2.3.5); pinned in requirements.txt.
- **MLflow** uses a local SQLite store (`mlflow.db`); the file store is deprecated in current MLflow.
- **Smoke vs trained separation.** `weights/manifest.json` carries `kind`; evaluation and
  benchmark write smoke output to `reports/eval_smoke/` / `reports/benchmark_smoke.*`, and
  `make report` only reads trained outputs, so smoke numbers cannot reach the README.
- **Model selection uses validation mAP50-95**, never the test split (`scripts/install_weights.py`).
- **Square accuracy counts empty squares** (64 per board), so a model that predicts nothing still
  scores ~50%; the headline metrics are boards with 0 / <=1 errors and mean errors per board.
