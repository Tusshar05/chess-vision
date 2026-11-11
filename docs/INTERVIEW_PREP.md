# Interview prep — 30 questions, answered from what was actually built

Rule: quote model-accuracy numbers only from `reports/` after you have trained. Facts about the
data/code below are verified in the repo.

## Object detection & modelling
**1. Why a detector rather than classifying each square?**
Pieces are tall: in perspective the top of a piece overlaps the next square, so a 64-crop classifier
sees ambiguous crops, and it needs perfect corners first. A detector localises each piece, and the
box *base* gives the square. It also reuses the same pretrained backbone and handles occlusion
gracefully (a missing detection is one wrong square, not a corrupted crop grid).

**2. Why YOLO11n?** 2.6 M params / ~6.5 GFLOPs at 640 px — it must run on CPU/edge. I did not assume
it is enough: the Colab notebook also trains 11s and imgsz 960 so the accuracy/latency trade-off is
measured (`reports/training/runs.md`).

**3. How do you evaluate the detector and what is mAP50-95?**
Ultralytics validation on the held-out test split: mAP50 (AP at IoU 0.5), mAP50-95 (mean over IoU
0.5…0.95 — rewards tight boxes), per-class P/R, and a confusion matrix to see look-alikes
(bishop/pawn, queen/king). But box mAP isn't the product metric — board-level accuracy is, so I
report both.

**4. What's NMS and why per-class?**
Non-max suppression drops overlapping boxes for the same object, keeping the highest score above
an IoU threshold. Per-class (offset boxes by class id) so a white bishop never suppresses a
neighbouring black pawn; remaining duplicates are handled at the square level (highest confidence wins).

**5. Why did you convert corner labels to image-space order?**
In ChessReD, `top_left` means the a8 corner, and in 834/834 checked images the a8 piece is nearest
it — but only 522 of 2,078 images have that corner at the image's top-left (photos from all four
sides). Making a keypoint model output "a8 corner" forces it to understand orientation. I train on
canonical image order and resolve orientation afterwards from piece colours. It also makes horizontal-flip augmentation correct.

**6. How is the learned corner model better (or not) than classical CV?**
I implemented an OpenCV baseline (inner-corner detector, else largest quad + calibrated margin) and
compare both on the same test images: corner error as % of board diagonal, % within 2 %/5 %, and
downstream board accuracy (`corner_comparison.json`). I do not pre-judge it — on early smoke-run
images the classical baseline was already decent, which is why it stays as a fallback.

**7. What would you change if bishops/pawns are confused?**
Higher imgsz (the 960 run), more augmentation for angle, per-piece re-classification on a tight crop
at higher resolution, and class-weighting; check whether errors concentrate at steep angles using the error-analysis tables.

## Geometry
**8. Explain a homography.**
A 3×3 matrix mapping points between two views of the same plane in homogeneous coordinates
(x' ∼ Hx, 8 DoF). Four point pairs determine it; `cv2.getPerspectiveTransform` solves the linear
system. The board is planar, so the 4 corners map the photo to a top-down grid.

**9. Why use the bottom of the box, not the centre?**
The homography is only valid for points *on the board plane*. The foot of a piece is on it; its centre
is above it and, in perspective, projects onto the wrong square. With ground-truth boxes and corners:
100 % perfect boards using a 10 %-lifted base vs 98.7 % plain bottom-centre (`geometry_oracle.json`).
I flag honestly that I picked the 0.10 among three values on the same images.

**10. How do you handle board orientation?**
Corners are in image-space order, so the a8 corner is unknown. I score the four rotations by "white
pieces on low ranks, black on high" (kings/pawns weighted ×2). Oracle accuracy 96.4 %; it fails on
6–7-piece endgames, so there's a `rotation` override. Alternative: a learned head or reading the
printed coordinates — extra labels/complexity.

**11. How do you deal with conflicts and impossible positions?**
Two boxes in one square → keep higher confidence and warn. Boxes outside the board (captured
pieces) are dropped. python-chess validation flags ≠ 1 king per side, pawns on rank 1/8, > 8 pawns, > 16
pieces, side-not-to-move in check. The UI makes the FEN editable so a human fixes residual errors.

**12. What can a photo not tell you?**
Side to move, castling rights, en passant, move counters. Turn is a parameter; castling is inferred
only when king and rook are on their home squares; en passant is never set. Video would help (temporal context).

## Data & evaluation
**13. What is data leakage and how did you avoid it?**
Consecutive photos in a game are near-identical, so a random image split puts almost the same board
in train and test and inflates scores. The official split is by game (60/20/20 games); an assertion
re-checks that no game appears in two splits on every run.

**14. Which metric would you put on the resume and why?**
Boards with zero errors (and ≤ 1 error, mean errors/board) on the game-disjoint test set — it is what a
user experiences. Square accuracy flatters: empties count, so predicting nothing scores ≈ 50 %.

**15. How does your result compare with the paper?**
The ChessReD paper reports 15.26 % exact boards (end-to-end CNN, full 2,129-image test split). Mine
is a detection+geometry pipeline on the 306-image ChessReD2K test subset (the part with box/corner
labels), so it's indicative, not controlled. I say so in the README.

**16. How did you choose the best model without touching the test set?**
`install_weights.py` selects by *validation* mAP50-95; the test split is only used for reporting.
INT8 calibration uses train/val images, not test.

**17. How do you know the pipeline works before training?**
A tiny smoke training (2 epochs, 40 images) produces weights flagged `kind: smoke`; the whole system
(export, evaluation, benchmark, API, Docker) is exercised with them. Their output goes to
`reports/eval_smoke/` and `make report` ignores it — so there is no way to publish smoke numbers by accident.

**18. What did the error analysis cover?**
Per camera, viewing angle (board foreshortening ratio as proxy), piece count (occlusion proxy), corner
source, orientation correctness, top confusions, and a gallery of worst boards.

## Optimisation & edge
**19. What is quantization and what did you do?**
Representing weights/activations in INT8 instead of FP32 → 4× smaller and potentially faster. I used
ONNX Runtime static quantization (QDQ, per-channel), calibrated with 50 train/val images to set
activation ranges. I measure latency *and* accuracy change; on some CPUs QDQ INT8 isn't faster.

**20. Static vs dynamic quantization?**
Dynamic computes activation scales at runtime (good for transformers/LSTMs, poor for convnets);
static fixes scales from calibration data and quantizes conv layers end-to-end.

**21. Tell me about a performance bug you found.**
My first benchmark showed a 53 ms resize for a 3072 px photo. Stage timing showed OpenCV was
single-threaded: importing Ultralytics calls `cv2.setNumThreads(0)`. The API image has no Ultralytics
so it was unaffected; I re-enabled threads in the benchmark to measure what's deployed.

**22. How would you run this on a phone / Jetson?**
iPhone: CoreML `.mlpackage` exported by `make export`, port the decode/NMS/homography (~100 lines).
Jetson: ONNX → TensorRT FP16 engine (`trtexec`), swap the ORT provider — *documented, untested*
(`docs/EDGE.md`), and I say so.

**23. Why no PyTorch in the serving image?**
Smaller image, faster start, fewer CVEs, and ONNX Runtime is faster to deploy; decoding/NMS are
re-implemented in NumPy/OpenCV and unit-tested against synthetic tensors.

## Services & engineering
**24. Walk me through the API design.**
`POST /v1/recognize` (multipart image → FEN, per-square confidences, warnings, annotated + top-down
images), `/v1/analyze` (FEN → Stockfish), `/v1/recognize-and-analyze`, `/v1/history`, `/health`
(reports smoke vs trained), `/metrics`. Pydantic schemas, a uniform error envelope, 415/413/422
validation, request IDs, JSON logs, Prometheus metrics, versioned `/v1` prefix.

**25. Why sync endpoints?**
Inference is CPU-bound; sync handlers run in Starlette's threadpool and ONNX Runtime sessions are
thread-safe. `async def` would block the event loop unless I used an executor anyway.

**26. How do you test the API without model files or Stockfish?**
Dependency injection: `create_app(recognizer=..., analyse_fn=...)`; tests inject fake detector/corner
finder and a fake engine, plus a TemporaryDirectory SQLite DB. One marked `e2e` test uses real files
and is skipped in CI.

**27. How does CI/CD work?**
GitHub Actions: ruff, pytest with coverage (models mocked; Stockfish installed via apt), Docker
buildx matrix for API and dashboard, push to GHCR on `main` with `GITHUB_TOKEN`. I ran the same
steps locally in a clean venv first.

**28. How does this scale?**
Recognition is stateless → add API replicas (HPA on CPU). The limit is SQLite history, a single-writer
file; for multi-replica use Postgres (`DB_URL`). Models are read-only on a PVC. Batch/queue only if
throughput requires; latency per image is tens of ms on CPU.

## Judgement
**29. Biggest limitations?**
Whole board must be visible; heavy occlusion and extreme angles; orientation ambiguity in sparse
endgames; single board/piece set (domain gap); look-alike pieces. Evaluated on one dataset only.

**30. What would you do next?**
Video: track the position across frames and accept only legal moves (temporal consistency → PGN
recording), a learned orientation head, hard-example mining from the failure gallery, an iPhone app,
Postgres history, and testing on photos of other boards/piece sets to measure the domain gap.
