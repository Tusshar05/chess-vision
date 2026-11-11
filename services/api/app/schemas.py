"""Pydantic request/response models (these also drive the OpenAPI docs)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SquareInfo(BaseModel):
    """A piece found on a square."""

    piece: str = Field(
        description="FEN symbol: uppercase = white, lowercase = black", examples=["N"]
    )
    confidence: float = Field(ge=0, le=1)


class DetectionOut(BaseModel):
    """One detected piece box in original-image pixels."""

    box: list[float] = Field(min_length=4, max_length=4, description="x1, y1, x2, y2")
    piece: str
    confidence: float


class ModelInfo(BaseModel):
    """Which model produced a result."""

    kind: str = Field(description="'smoke' (pipeline test, inaccurate) or 'trained'")
    detector: str | None = None
    corners: str | None = None
    precision: str


class LineOut(BaseModel):
    """One engine principal variation."""

    move: str = Field(description="first move, UCI")
    san: str
    score_cp: int | None = Field(description="centipawns, White's point of view")
    mate: int | None = Field(description="moves to mate, White's POV; negative = Black mates")
    pv: list[str] = Field(description="variation in SAN")


class AnalysisOut(BaseModel):
    """Stockfish analysis of a position."""

    fen: str
    best_move: str | None = Field(description="UCI")
    best_move_san: str | None
    score_cp: int | None
    mate: int | None
    lines: list[LineOut]
    depth: int | None
    game_over: bool = False


class AnalyzeRequest(BaseModel):
    """Request body for ``POST /v1/analyze``."""

    fen: str = Field(
        min_length=10,
        max_length=100,
        examples=["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
    )
    depth: int = Field(default=14, ge=1, le=30)
    time_limit: float | None = Field(
        default=None, gt=0, le=30, description="seconds; overrides depth"
    )
    multipv: int = Field(default=3, ge=1, le=5)


class RecognitionOut(BaseModel):
    """Result of ``POST /v1/recognize``."""

    id: int
    fen: str
    turn: Literal["w", "b"]
    rotation: int = Field(ge=0, le=3, description="0 = white at the bottom of the canonical view")
    squares: dict[str, SquareInfo]
    warnings: list[str]
    corners: list[list[float]] = Field(description="4 board corners in image pixels, clockwise")
    corner_source: str
    corner_confidence: float
    detections: list[DetectionOut]
    timings_ms: dict[str, float]
    model: ModelInfo
    annotated_image: str | None = Field(default=None, description="base64 JPEG with boxes/outline")
    board_image: str | None = Field(default=None, description="base64 JPEG top-down board")
    board_svg: str | None = Field(default=None, description="SVG render of the recognised FEN")


class RecognizeAndAnalyzeOut(RecognitionOut):
    """Recognition plus Stockfish analysis (``analysis`` is null if the position is unusable)."""

    analysis: AnalysisOut | None = None
    analysis_error: str | None = None


class HistoryItem(BaseModel):
    """A stored past recognition."""

    id: int
    created_at: datetime
    fen: str
    turn: str
    filename: str | None
    warnings: list[str]
    corner_source: str
    model_kind: str
    latency_ms: float


class HistoryOut(BaseModel):
    """Paged history."""

    total: int
    items: list[HistoryItem]


class HealthOut(BaseModel):
    """Service health, including which model is loaded."""

    status: Literal["ok", "degraded"]
    version: str
    model: ModelInfo
    stockfish_available: bool


class ErrorBody(BaseModel):
    """Uniform error payload."""

    code: str
    message: str


class ErrorOut(BaseModel):
    """Error envelope."""

    error: ErrorBody
