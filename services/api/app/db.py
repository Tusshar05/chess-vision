"""SQLite persistence for recognition history (SQLAlchemy 2.x)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    """Declarative base."""


class Recognition(Base):
    """One stored recognition (no image bytes are kept, only the result)."""

    __tablename__ = "recognitions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    fen: Mapped[str] = mapped_column(String(120))
    turn: Mapped[str] = mapped_column(String(1))
    warnings_json: Mapped[str] = mapped_column(Text, default="[]")
    corner_source: Mapped[str] = mapped_column(String(16))
    model_kind: Mapped[str] = mapped_column(String(16))
    latency_ms: Mapped[float] = mapped_column(Float)


class History:
    """Thin repository around the table."""

    def __init__(self, url: str):
        kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, **kwargs)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def add(
        self,
        *,
        filename: str | None,
        fen: str,
        turn: str,
        warnings: list[str],
        corner_source: str,
        model_kind: str,
        latency_ms: float,
    ) -> int:
        """Insert a recognition and return its id."""
        with self.Session.begin() as s:
            row = Recognition(
                filename=filename,
                fen=fen,
                turn=turn,
                warnings_json=json.dumps(warnings),
                corner_source=corner_source,
                model_kind=model_kind,
                latency_ms=latency_ms,
            )
            s.add(row)
            s.flush()
            return row.id

    def list(self, limit: int, offset: int) -> tuple[int, list[Recognition]]:
        """Newest-first page plus the total count."""
        with self.Session() as s:
            total = s.scalar(select(func.count()).select_from(Recognition)) or 0
            rows = s.scalars(
                select(Recognition).order_by(Recognition.id.desc()).limit(limit).offset(offset)
            ).all()
            return total, list(rows)
