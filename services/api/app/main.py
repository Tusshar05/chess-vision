"""FastAPI application: recognise a chessboard photo, analyse positions, keep history."""

from __future__ import annotations

import base64
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from ml.chessvision import engine as sf
from ml.chessvision import render
from ml.chessvision.constants import CLASS_TO_SYMBOL
from ml.chessvision.pipeline import BoardNotFoundError, Recognizer, load_recognizer, read_manifest
from services.api.app import __version__, logging_utils
from services.api.app.config import Settings
from services.api.app.db import History
from services.api.app.schemas import (
    AnalysisOut,
    AnalyzeRequest,
    DetectionOut,
    ErrorOut,
    HealthOut,
    HistoryItem,
    HistoryOut,
    LineOut,
    ModelInfo,
    RecognitionOut,
    RecognizeAndAnalyzeOut,
    SquareInfo,
)

log = logging.getLogger("chessvision.api")
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_PIXELS = 100_000_000


class ApiError(Exception):
    """An error with an HTTP status and a stable machine-readable code."""

    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


AnalyseFn = Callable[..., sf.Analysis]


def create_app(
    settings: Settings | None = None,
    recognizer: Recognizer | None = None,
    analyse_fn: AnalyseFn | None = None,
) -> FastAPI:
    """Build the app. ``recognizer`` / ``analyse_fn`` can be injected (tests use fakes)."""
    cfg = settings or Settings()
    logging_utils.configure(cfg.log_level)
    registry = CollectorRegistry()
    m_req = Counter(
        "chessvision_requests_total",
        "HTTP requests",
        ["path", "method", "status"],
        registry=registry,
    )
    m_lat = Histogram("chessvision_request_seconds", "Request latency", ["path"], registry=registry)
    m_rec = Counter(
        "chessvision_recognitions_total", "Recognitions", ["model_kind"], registry=registry
    )
    m_stage = Histogram(
        "chessvision_stage_ms",
        "Pipeline stage latency (ms)",
        ["stage"],
        buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
        registry=registry,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = cfg
        app.state.history = History(cfg.db_url)
        app.state.manifest = read_manifest(Path(cfg.weights_dir))
        app.state.analyse = analyse_fn or sf.analyse
        if recognizer is not None:
            app.state.recognizer = recognizer
        else:
            try:
                app.state.recognizer = load_recognizer(
                    cfg.weights_dir, cfg.precision, conf=cfg.detect_conf
                )
            except Exception as e:  # noqa: BLE001 - serve /health even without models
                log.error("model load failed", extra={"error": str(e)})
                app.state.recognizer = None
        log.info(
            "api started",
            extra={
                "model_kind": app.state.manifest.get("kind"),
                "models_loaded": app.state.recognizer is not None,
            },
        )
        yield

    app = FastAPI(
        title="ChessVision API",
        version=__version__,
        description="Photo of a chessboard -> FEN -> Stockfish. Models run on ONNX Runtime (CPU).",
        lifespan=lifespan,
        responses={422: {"model": ErrorOut}, 503: {"model": ErrorOut}},
    )

    # ---- middleware / errors -------------------------------------------------
    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = rid
            return response
        finally:
            dt = time.perf_counter() - t0
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            m_req.labels(path, request.method, str(status)).inc()
            m_lat.labels(path).observe(dt)
            log.info(
                "request",
                extra={
                    "request_id": rid,
                    "method": request.method,
                    "path": path,
                    "status": status,
                    "ms": round(dt * 1000, 1),
                },
            )

    def err(status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(
            status_code=status, content={"error": {"code": code, "message": message}}
        )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError):
        return err(e.status, e.code, e.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        first = e.errors()[0] if e.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", []))
        return err(422, "validation_error", f"{loc}: {first.get('msg', 'invalid request')}")

    @app.exception_handler(HTTPException)
    async def _http(_: Request, e: HTTPException):
        return err(e.status_code, "http_error", str(e.detail))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, e: Exception):
        log.exception("unhandled error")
        return err(500, "internal_error", "unexpected server error")

    # ---- helpers ---------------------------------------------------------------
    def get_rec(request: Request) -> Recognizer:
        rec = request.app.state.recognizer
        if rec is None:
            raise ApiError(503, "models_unavailable", "model files could not be loaded")
        return rec

    def model_info(request: Request) -> ModelInfo:
        m = request.app.state.manifest
        return ModelInfo(
            kind=m.get("kind", "unknown"),
            detector=m.get("detector"),
            corners=m.get("corners"),
            precision=cfg.precision,
        )

    def read_image(file: UploadFile) -> np.ndarray:
        if file.content_type not in ALLOWED_TYPES:
            raise ApiError(
                415,
                "unsupported_media_type",
                f"expected one of {sorted(ALLOWED_TYPES)}, got {file.content_type}",
            )
        data = file.file.read(cfg.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > cfg.max_upload_mb * 1024 * 1024:
            raise ApiError(413, "file_too_large", f"max upload is {cfg.max_upload_mb} MB")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ApiError(422, "invalid_image", "file could not be decoded as an image")
        if img.shape[0] * img.shape[1] > MAX_PIXELS:
            raise ApiError(413, "image_too_large", "image has too many pixels")
        return img

    def b64_jpeg(img: np.ndarray, max_side: int = 1280, q: int = 80) -> str:
        s = min(1.0, max_side / max(img.shape[:2]))
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return base64.b64encode(buf.tobytes()).decode()

    def do_recognize(
        request: Request, file: UploadFile, rotation: int | None, turn: str, include_images: bool
    ) -> RecognitionOut:
        rec = get_rec(request)
        img = read_image(file)
        try:
            res = rec.recognize(img, rotation=rotation, turn=turn)
        except BoardNotFoundError as e:
            raise ApiError(422, "board_not_found", str(e)) from e
        for stage, ms in res.timings_ms.items():
            if stage != "total":
                m_stage.labels(stage).observe(ms)
        info = model_info(request)
        m_rec.labels(info.kind).inc()
        rid = request.app.state.history.add(
            filename=file.filename,
            fen=res.fen,
            turn=turn,
            warnings=res.warnings,
            corner_source=res.corner_source,
            model_kind=info.kind,
            latency_ms=res.timings_ms["total"],
        )
        if info.kind == "smoke":
            res.warnings.append("SMOKE-TEST model loaded: output is not meaningful")
        out = RecognitionOut(
            id=rid,
            fen=res.fen,
            turn=turn,
            rotation=res.rotation,
            squares={sq: SquareInfo(**v) for sq, v in res.square_confidences().items()},
            warnings=res.warnings,
            corners=[[round(float(x), 1), round(float(y), 1)] for x, y in res.corners],
            corner_source=res.corner_source,
            corner_confidence=round(res.corner_conf, 4),
            detections=[
                DetectionOut(
                    box=[round(v, 1) for v in (d.x1, d.y1, d.x2, d.y2)],
                    piece=CLASS_TO_SYMBOL[d.cls],
                    confidence=round(d.conf, 4),
                )
                for d in res.detections
            ],
            timings_ms=res.timings_ms,
            model=info,
        )
        if include_images:
            out.annotated_image = b64_jpeg(render.annotate(img, res.corners, res.detections))
            out.board_image = b64_jpeg(render.top_down(img, res.corners, res.reconstruction), 640)
            out.board_svg = render.board_svg(res.fen)
        return out

    def do_analyze(
        request: Request, fen: str, depth: int, time_limit: float | None, multipv: int
    ) -> AnalysisOut:
        try:
            a = request.app.state.analyse(fen, depth=depth, time_limit=time_limit, multipv=multipv)
        except sf.EngineUnavailableError as e:
            raise ApiError(503, "engine_unavailable", str(e)) from e
        except ValueError as e:
            raise ApiError(422, "invalid_position", str(e)) from e
        return AnalysisOut(
            fen=a.fen,
            best_move=a.best_move,
            best_move_san=a.best_move_san,
            score_cp=a.score_cp,
            mate=a.mate,
            depth=a.depth,
            game_over=a.game_over,
            lines=[
                LineOut(move=ln.move, san=ln.san, score_cp=ln.score_cp, mate=ln.mate, pv=ln.pv)
                for ln in a.lines
            ],
        )

    # ---- routes ----------------------------------------------------------------
    @app.post(
        "/v1/recognize",
        response_model=RecognitionOut,
        tags=["recognition"],
        summary="Photo of a board -> FEN",
    )
    def recognize(
        request: Request,
        file: UploadFile = File(description="JPEG/PNG/WebP photo of the board"),
        rotation: int | None = Form(
            default=None, ge=0, le=3, description="override board orientation (0 = white at bottom)"
        ),
        turn: str = Form(default="w", pattern="^[wb]$", description="side to move"),
        include_images: bool = Form(default=True),
    ):
        return do_recognize(request, file, rotation, turn, include_images)

    @app.post(
        "/v1/analyze",
        response_model=AnalysisOut,
        tags=["analysis"],
        summary="FEN -> Stockfish best move, evaluation, top lines",
    )
    def analyze(request: Request, body: AnalyzeRequest):
        return do_analyze(request, body.fen, body.depth, body.time_limit, body.multipv)

    @app.post(
        "/v1/recognize-and-analyze",
        response_model=RecognizeAndAnalyzeOut,
        tags=["recognition", "analysis"],
        summary="Photo -> FEN -> Stockfish in one call",
    )
    def recognize_and_analyze(
        request: Request,
        file: UploadFile = File(...),
        rotation: int | None = Form(default=None, ge=0, le=3),
        turn: str = Form(default="w", pattern="^[wb]$"),
        include_images: bool = Form(default=True),
        depth: int = Form(default=cfg.default_depth, ge=1, le=30),
        time_limit: float | None = Form(default=None, gt=0, le=30),
        multipv: int = Form(default=3, ge=1, le=5),
    ):
        base = do_recognize(request, file, rotation, turn, include_images)
        out = RecognizeAndAnalyzeOut(**base.model_dump())
        try:
            out.analysis = do_analyze(request, base.fen, depth, time_limit, multipv)
            if out.board_svg and out.analysis.best_move:
                out.board_svg = render.board_svg(base.fen, out.analysis.best_move)
        except ApiError as e:  # recognition succeeded; analysis can fail on an implausible board
            out.analysis_error = f"{e.code}: {e.message}"
        return out

    @app.get("/v1/history", response_model=HistoryOut, tags=["history"])
    def history(
        request: Request, limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)
    ):
        import json

        total, rows = request.app.state.history.list(limit, offset)
        return HistoryOut(
            total=total,
            items=[
                HistoryItem(
                    id=r.id,
                    created_at=r.created_at,
                    fen=r.fen,
                    turn=r.turn,
                    filename=r.filename,
                    warnings=json.loads(r.warnings_json),
                    corner_source=r.corner_source,
                    model_kind=r.model_kind,
                    latency_ms=r.latency_ms,
                )
                for r in rows
            ],
        )

    @app.get("/health", response_model=HealthOut, tags=["ops"])
    def health(request: Request):
        try:
            sf.find_stockfish()
            have_sf = True
        except sf.EngineUnavailableError:
            have_sf = False
        loaded = request.app.state.recognizer is not None
        return HealthOut(
            status="ok" if loaded and have_sf else "degraded",
            version=__version__,
            model=model_info(request),
            stockfish_available=have_sf,
        )

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


def app_factory() -> FastAPI:
    """uvicorn entry point: ``uvicorn services.api.app.main:app_factory --factory``."""
    return create_app()
