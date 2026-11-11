"""Shared constants: class ids, FEN symbols, square naming."""

from __future__ import annotations

# Order matches ChessReD category ids 0..11 and the YOLO class ids used for training.
CLASS_NAMES = [
    "white-pawn", "white-rook", "white-knight", "white-bishop", "white-queen", "white-king",
    "black-pawn", "black-rook", "black-knight", "black-bishop", "black-queen", "black-king",
]  # fmt: skip
CLASS_TO_SYMBOL = dict(enumerate("PRNBQKprnbqk"))
SYMBOL_TO_CLASS = {s: i for i, s in CLASS_TO_SYMBOL.items()}
FILES = "abcdefgh"
