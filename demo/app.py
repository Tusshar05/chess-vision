"""Gradio demo: photo -> FEN, board render, Stockfish best move (ONNX Runtime, CPU).

Runs locally (``python demo/app.py``) or on Hugging Face Spaces (see docs/DEPLOY_DEMO.md).
Models are read from ``$WEIGHTS_DIR`` (default ``weights`` locally, ``models`` on Spaces).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.chessvision import engine, render  # noqa: E402
from ml.chessvision.pipeline import BoardNotFoundError, load_recognizer, read_manifest  # noqa: E402

HERE = Path(__file__).resolve().parent
WEIGHTS = Path(
    os.environ.get("WEIGHTS_DIR", HERE / "models" if (HERE / "models").exists() else "weights")
)
MANIFEST = read_manifest(WEIGHTS)
RECOGNIZER = load_recognizer(WEIGHTS, precision=os.environ.get("MODEL_PRECISION", "fp32"))
SAMPLES = sorted((HERE / "samples").glob("*.jpg")) or sorted(Path("samples").glob("*.jpg"))


def run(image: np.ndarray | None, turn: str, rotation: str, depth: int):
    """Gradio callback. ``image`` is RGB from Gradio."""
    if image is None:
        raise gr.Error("Upload a photo of a chessboard first.")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    try:
        res = RECOGNIZER.recognize(
            bgr,
            rotation=None if rotation == "auto" else int(rotation),
            turn="w" if turn == "White" else "b",
        )
    except BoardNotFoundError as e:
        raise gr.Error(f"{e}. Try a clearer, more top-down photo.") from e
    ann = cv2.cvtColor(render.annotate(bgr, res.corners, res.detections), cv2.COLOR_BGR2RGB)
    top = cv2.cvtColor(render.top_down(bgr, res.corners, res.reconstruction), cv2.COLOR_BGR2RGB)
    best, move = "Analysis unavailable for this position.", None
    try:
        a = engine.analyse(res.fen, depth=int(depth), multipv=3)
        if a.game_over:
            best = "Game over in this position."
        else:
            ev = (
                f"mate in {abs(a.mate)}"
                if a.mate is not None
                else f"{(a.score_cp or 0) / 100:+.2f}"
            )
            best = f"Best move: {a.best_move_san}   (eval {ev}, depth {a.depth})\n" + "\n".join(
                f"  {ln.san}: {' '.join(ln.pv[:6])}" for ln in a.lines
            )
            move = a.best_move
    except (ValueError, engine.EngineUnavailableError) as e:
        best = f"Stockfish skipped: {e}"
    svg = str(render.board_svg(res.fen, move, size=420))
    warn = "\n".join(f"- {w}" for w in res.warnings) or "none"
    return ann, top, res.fen, svg, best, warn


with gr.Blocks(title="ChessVision") as demo:
    gr.Markdown("# ♟️ ChessVision — photo to FEN to best move")
    if MANIFEST.get("kind") != "trained":
        gr.Markdown("⚠️ **Smoke-test model loaded: results are not meaningful.**")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Photo of a chessboard")
            turn = gr.Radio(["White", "Black"], value="White", label="Side to move")
            rot = gr.Dropdown(["auto", "0", "1", "2", "3"], value="auto", label="Board orientation")
            depth = gr.Slider(6, 20, value=12, step=1, label="Stockfish depth")
            go = gr.Button("Recognise", variant="primary")
            if SAMPLES:
                gr.Examples(
                    [[str(p)] for p in SAMPLES], inputs=[inp], label="Samples (ChessReD test split)"
                )
        with gr.Column():
            fen = gr.Textbox(label="FEN", interactive=False)
            board = gr.HTML(label="Recognised board")
            best = gr.Textbox(label="Stockfish", lines=4)
            warn = gr.Markdown()
    with gr.Row():
        out_ann = gr.Image(label="Detections")
        out_top = gr.Image(label="Top-down view")
    go.click(run, [inp, turn, rot, depth], [out_ann, out_top, fen, board, best, warn])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
