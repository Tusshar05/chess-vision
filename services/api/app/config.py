"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """API configuration (all overridable through the environment)."""

    weights_dir: str = field(default_factory=lambda: os.environ.get("WEIGHTS_DIR", "weights"))
    precision: str = field(default_factory=lambda: os.environ.get("MODEL_PRECISION", "fp32"))
    db_url: str = field(default_factory=lambda: os.environ.get("DB_URL", "sqlite:///./history.db"))
    max_upload_mb: int = field(default_factory=lambda: int(os.environ.get("MAX_UPLOAD_MB", "15")))
    detect_conf: float = field(default_factory=lambda: float(os.environ.get("DETECT_CONF", "0.25")))
    default_depth: int = field(default_factory=lambda: int(os.environ.get("ENGINE_DEPTH", "14")))
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
