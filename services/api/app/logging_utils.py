"""Structured (JSON) logging."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """One JSON object per log line; ``extra={...}`` fields are merged in."""

    RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message"}

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as JSON."""
        d = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        d.update({k: v for k, v in record.__dict__.items() if k not in self.RESERVED})
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, default=str)


def configure(level: str = "INFO") -> None:
    """Route the root logger to stdout as JSON (idempotent)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_chessvision", False):
            root.removeHandler(h)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    h._chessvision = True  # type: ignore[attr-defined]
    root.addHandler(h)
    root.setLevel(level)
