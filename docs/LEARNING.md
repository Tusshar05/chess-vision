# LEARNING.md — understand the whole system

Goal: you can explain every component, why it was chosen, what the alternatives were, and follow
one photo from pixels to a Stockfish move. Numbers quoted here are facts about the data or the
code; model-accuracy numbers are deliberately absent until you have trained (see `reports/`).

## 0. The one-paragraph version
A phone photo goes through (1) a corner model that finds the four corners of the board and (2) a
detector that finds every piece with a box and a class (12 classes). The corners define a
**homography** that maps the image plane to a top-down 8×8 grid; the **base of each piece's box**
is projected into that grid to find its square. Conflicts are resolved by confidence, the board
**orientation** is inferred from where the white and black pieces are, **python-chess** builds a
FEN and checks it is plausible, and **Stockfish** analyses it. All inference is ONNX Runtime on CPU.

## 1. Data flow, step by step (`ml/chessvision/pipeline.py`)
| # | Stage | Code | What happens |
|---|---|---|---|
| 1 | Decode + validate | `services/api/app/main.py: read_image` | content type, size ≤ 15 MB, decodes, ≤ 100 MP |
| 2 | Downscale | `pipeline.downscale` | area-resize to ≤ 1280 px long side (same as training data prep) |
| 3 | Letterbox | `onnx_infer.letterbox` | resize to 640×640 keeping aspect ratio, grey padding, RGB, /255 |
| 4 | Corner model | `OnnxCornerModel.find` | YOLO11n-pose forward pass → best board's 4 keypoints |
| 5 | Piece detector | `OnnxDetector.detect` | forward pass → raw tensor → threshold → per-class NMS |
| 6 | Rescale | `pipeline.recognize` | boxes/corners back to original-photo pixels |
| 7 | Homography | `board.homography` | `cv2.getPerspectiveTransform` from 4 corners to an 800×800 square |
| 8 | Square assignment | `board.locate_cells`, `build_squares` | project each box's anchor; cell = floor(u/100), floor(v/100) |
| 9 | Orientation | `board.resolve_orientation` | try 4 rotations, keep the one with white low / black high |
| 10 | FEN + checks | `board.build_board`, `validate_board` | python-chess; warnings for impossible positions |
| 11 | Engine | `engine.analyse` | Stockfish over UCI: best move, score, top-3 lines |

## 2. Components and the reasoning behind them

### 2.1 Dataset: ChessReD
10,800 phone photos, 100 games, 3 phones, piece-per-square labels for everything; **boxes and
corners only for the 2,078-image ChessReD2K subset** (20 games). Official split by game:
60/20/20 games → 6,479 / 2,192 / 2,129 images. Citation in the README.
- **Why split by game, not by image?** Consecutive photos in a game differ by one move: the board
  and lighting are near-identical. A random image split puts near-duplicates in train and test and
  inflates the score (data leakage). `Annotations.assert_no_game_leakage` enforces it.
- **License trap:** the brief said CC BY 4.0; the source record says CC BY-NC-SA 4.0. I followed
  the source and documented it (`docs/DECISIONS.md`). Interview point: *verify licenses at the source*.
- **Sizes:** full archive 24.6 GB vs 12 GB of free disk → `fetch_from_remote_zip` reads the zip's
  central directory over HTTP Range requests and downloads single JPEGs. Colab pulls the 4.7 GB
  ChessReD2K archive with the same module.

### 2.2 Label conversion (`ml/data/chessred.py`)
- Detection labels: COCO `[x,y,w,h]` pixels → YOLO `class cx cy w h` normalised. 12 classes
  (category 12 "empty" is not an object).
- Corner labels → YOLO-pose: one class "board", 4 keypoints, box = padded hull of the corners.
- **Finding:** the dataset's corner names are *board-relative* (`top_left` = a8 corner) — in 834/834
  checked images the a8 piece is nearest `top_left` — but only 522/2078 have it at the image's top-left,
  because photos are taken from all four sides. A pose model asked to output "the a8 corner" must
  recognise board orientation, which is hard and unnecessary. **Decision:** train on *image-space
  canonical order* (start at the corner with min x+y, then clockwise; `canonical_corners`), and find
  orientation afterwards from the pieces. It also makes `fliplr` augmentation valid
  (`flip_idx=[1,0,3,2]`).
- Images are stored at ≤ 1280 px: smaller files, faster epochs, and still enough for imgsz 960.

### 2.3 Models: YOLO11n (+ pose variant)
- **Why YOLO, why detection?** Pieces are small, tall objects on a regular grid with occlusion.
  A single-stage detector gives boxes + classes in one pass; the box *base* gives position.
  Alternatives: (a) classify each of the 64 warped squares with a CNN (needs good corners first;
  tall pieces overflow into neighbouring squares in perspective, so square crops are ambiguous);
  (b) end-to-end board prediction like the ChessReD paper (their model gets 15.26 % exact boards);
  (c) segmentation (more labels, no gain).
- **Why nano?** 2.6 M parameters, ~6.5 GFLOPs @ 640 → CPU/edge-friendly. We also train 11s and
  imgsz 960 to *measure* the accuracy/latency trade-off instead of guessing.
- **Corner model as pose:** four keypoints of one object is exactly YOLO-pose. Alternatives:
  heatmap regression, a segmentation mask + polygon fit, or classical CV (below).
- **Classical baseline (`corners_classic.py`):** `findChessboardCornersSB` (needs a clean board) →
  else largest quadrilateral in the Canny edge map (the printed mat) shrunk by a margin
  (0.079 of the mat) *measured on 36 train images* (`scripts/calibrate_classic_corners.py`). It exists
  as (a) a fallback and (b) an honest comparison for the learned model.
- **Training (`ml/train.py`):** config-driven YAML, MLflow (SQLite store), per-run `metrics.json`,
  wall-clock cap via Ultralytics' `time=`, resume from `last.pt`, skip finished runs.

### 2.4 Geometry (`ml/chessvision/board.py`) — the part interviewers love
- **Homography** = the 3×3 matrix H mapping points on one plane to another plane seen from a
  different viewpoint (`x' ~ Hx` in homogeneous coordinates; 8 degrees of freedom → 4 point
  correspondences). The board is planar, so the 4 corners determine it exactly.
- **Why the base of the box?** A piece is a 3-D object standing on a square. Projecting the box
  *centre* through the board homography would land on a different square for tall pieces seen from
  a low angle (the top of a king leans over the next square). The foot is on the board plane, where
  the homography is valid. I lift the anchor by 10 % of box height because the base ellipse sits
  slightly above the box's bottom edge: with ground-truth boxes+corners this gives 100 % perfect
  boards vs 98.7 % for the plain bottom-centre (`reports/eval/geometry_oracle.json`).
- **Conflicts:** two detections in one cell → keep the higher confidence. Detections outside the
  board (with 0.2-cell tolerance) are dropped: captured pieces sit beside the board in many photos.
- **Orientation:** the corner order is image-space, so which board corner is a8 is unknown. For each
  of 4 rotations score "white pieces on low ranks, black on high" (kings/pawns ×2). Oracle accuracy
  is 96.4 % — it fails on 6–7-piece endgames, which is why `rotation` is an API/UI parameter.
- **FEN:** python-chess `Board`; castling rights only when king and rook are on home squares;
  en passant never; side to move is a parameter (a photo can't know).
- **Validation:** exactly one king per side, ≤ 8 pawns, ≤ 16 pieces, no pawns on rank 1/8, side not
  to move not in check (python-chess `status()`).

### 2.5 Inference stack (`onnx_infer.py`)
- Ultralytics exports a *raw* head: detector `(1, 4+12, N)`, pose `(1, 5+12, N)`. Decoding and NMS
  are ~40 lines of NumPy + `cv2.dnn.NMSBoxes`; per-class NMS via class-offset boxes.
- **No PyTorch in the API image** → smaller image, faster cold start, simpler security surface.
- **INT8 (`ml/export.py`):** ONNX Runtime *static* quantization, QDQ format, per-channel weights,
  calibrated on 50 train/val images (never test). Dynamic quantization is poor for conv nets;
  static needs calibration data to set activation ranges. Whether INT8 is *faster* depends on the
  CPU — the benchmark measures it; do not assume.
- **CoreML:** `.mlpackage` for iPhone/Neural Engine. coremltools 9 crashed on numpy ≥ 2.3 → pinned.
- **Benchmark gotcha I hit:** importing Ultralytics calls `cv2.setNumThreads(0)`, making the 3072→1280
  resize ~7× slower (53 ms vs 7 ms). The API has no Ultralytics so it doesn't suffer; the
  benchmark re-enables threads to measure the deployed system. Lesson: *profile stage by stage*.

### 2.6 Services
- **FastAPI** (`services/api/app/main.py`): sync endpoints (CPU-bound work runs in Starlette's
  threadpool), Pydantic schemas = validation + OpenAPI, uniform error envelope
  `{"error": {"code", "message"}}`, request-id middleware, JSON logs, Prometheus counters/histograms
  (requests, recognitions, per-stage latency), `/health` says which model **kind** (smoke/trained) is loaded.
- **Persistence:** SQLAlchemy 2 + SQLite on a volume; stores results, not images (privacy, size).
- **Streamlit dashboard** talks to the API over HTTP only. Editable FEN (the human fixes the
  model's mistakes) → python-chess SVG → Stockfish → Lichess link.
- **Docker:** multi-stage, non-root (uid 10001), healthchecks, Stockfish from apt (separate GPL
  binary), weights as a bind-mounted volume so swapping models needs no rebuild.
- **Kubernetes:** Deployments/Services/ConfigMap/PVCs/HPA validated with kubeconform (not run on a
  cluster). SQLite + HPA caveat documented.
- **CI/CD:** ruff, pytest+coverage (models/Stockfish mocked → no large files in CI), Docker build
  matrix, GHCR push on main with `GITHUB_TOKEN`.

## 3. The Colab training setup (explain this confidently)
- **Why Colab:** the Mac has no NVIDIA GPU; a free T4 trains YOLO11n at 640 px in tens of minutes.
- **`notebooks/train_colab.ipynb`** is generated by `scripts/build_notebook.py` and has a
  `CHESSVISION_DRY_RUN=1` mode that runs the same cells locally (we ran it to validate it).
  Cells: GPU check (fails loudly without a GPU) → mount Drive → clone repo → pip install →
  `python -m ml.data.prepare --mode full` (download + MD5 verify + split-leakage check + YOLO
  conversion) → `python -m ml.colab_runner` (all runs) → metrics table → download zip.
- **Disconnect resilience:** all outputs go to Drive; Ultralytics writes `last.pt` every epoch;
  `train.run` resumes from it (`resume=True`) and skips runs that already have `metrics.json`.
- **Time budget:** each config has `time_hours` (0.7 / 0.5), a hard wall-clock cap; priorities:
  `det_n_640`, `corners_n_640`, `det_n_960`, `det_s_640`.
- **Output:** `chessvision_weights.zip` = best weights + each run's `metrics.json` + `manifest.json`
  with SHA-256. `scripts/install_weights.py` verifies checksums, checks each model's task, selects
  the detector by **validation** mAP50-95 (never test) and writes `weights/manifest.json`
  (`kind: trained`) — which flips `/health`, evaluation outputs and the README from smoke to real.
- **Smoke vs real separation** is deliberate: smoke runs write to `reports/eval_smoke/` (gitignored);
  `make report` reads only trained outputs.

## 4. Evaluation (`ml/evaluate.py`)
- **Detector:** mAP50, mAP50-95, per-class P/R, confusion matrix (watch bishop↔pawn).
- **End-to-end:** per board, errors = squares whose content differs (type, colour or occupancy).
  Reported: % boards with 0 errors, ≤ 1 error, mean errors/board, square accuracy (counts empties,
  so all-empty predictions score ≈ 50 % — headline is board-level).
- **Ablation rows:** auto vs given orientation (separates orientation from recognition errors);
  learned vs OpenCV corners (same detector).
- **Paper comparison:** their 15.26 % exact-board number is on 2,129 test images with a different
  method; ours uses the 306-image ChessReD2K test subset. Say "indicative".
- **Error analysis:** tables by camera, viewing angle (foreshortening ratio proxy), piece count,
  corner source, orientation outcome, plus the failure gallery and the most common square confusions.

## 5. Alternatives you should be able to discuss
| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Recognition | detector + homography | 64-square classifier; end-to-end CNN | handles occlusion/perspective, reuses boxes |
| Corners | learned pose + classical fallback | classical only; heatmap net | measured comparison |
| Orientation | piece-colour heuristic + override | learned head; board markings/OCR | no extra labels |
| Serving | ONNX Runtime CPU | PyTorch; TensorRT; TFLite | small image, portable |
| Quantization | static INT8 QDQ | dynamic; FP16 | accuracy/size trade-off measured |
| API | FastAPI sync + threadpool | async + executor; gRPC | simple, CPU-bound |
| Storage | SQLite | Postgres | single-node demo; documented upgrade |
| Engine | Stockfish subprocess | python port; cloud API | strength, licensing separation |

## 6. Things that went wrong (good interview stories)
1. Corner labels were board-relative, not image-relative → re-defined the keypoint target.
2. First benchmark had a 7× slow resize caused by an import side-effect (OpenCV threads).
3. A silent classical-corner fallback inflated INT8 latency → benchmark uses `learned_only`.
4. coremltools/numpy version clash → pinned.
5. MLflow file store deprecated → SQLite.
6. License mismatch between the brief and the data source.
