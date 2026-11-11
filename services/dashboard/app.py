"""Streamlit dashboard: upload a photo, fix the FEN, get Stockfish's view.

Talks to the API over HTTP only (``API_URL``), so it never loads a model itself.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from services.dashboard.helpers import (
    eval_text,
    fen_error,
    lichess_url,
    set_turn,
    svg_data_uri,
)

API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
SAMPLES = Path(os.environ.get("SAMPLES_DIR", "samples"))
TIMEOUT = 120

st.set_page_config(page_title="ChessVision", page_icon="♟️", layout="wide")


def api(method: str, path: str, **kw):
    """Call the API; surface its error envelope as a Streamlit error."""
    try:
        r = requests.request(method, f"{API}{path}", timeout=TIMEOUT, **kw)
    except requests.RequestException as e:
        st.error(f"Cannot reach the API at {API}: {e}")
        return None
    if r.status_code >= 400:
        try:
            e = r.json()["error"]
            st.error(f"{e['code']}: {e['message']}")
        except Exception:  # noqa: BLE001
            st.error(f"API error {r.status_code}")
        return None
    return r.json()


def show_b64(img_b64: str | None, caption: str) -> None:
    """Render a base64 JPEG."""
    if img_b64:
        st.image(base64.b64decode(img_b64), caption=caption, use_container_width=True)


st.title("♟️ ChessVision")
st.caption("Photo of a physical board → pieces → FEN → Stockfish")

health = api("GET", "/health")
if health:
    kind = health["model"]["kind"]
    if kind != "trained":
        st.error(
            "⚠️ **Smoke-test weights loaded — results not meaningful.** "
            "Train on Colab and run `make weights export` for real accuracy."
        )
    if not health["stockfish_available"]:
        st.error("Stockfish is not available on the API server.")

with st.sidebar:
    st.header("Settings")
    turn = st.radio(
        "Side to move",
        ["w", "b"],
        format_func=lambda t: {"w": "White", "b": "Black"}[t],
        horizontal=True,
    )
    rot_choice = st.selectbox(
        "Board orientation",
        ["auto", 0, 1, 2, 3],
        help="0 = white at the bottom of the canonical view. Use if auto is wrong.",
    )
    depth = st.slider("Stockfish depth", 6, 24, 14)
    lines = st.slider("Lines to show", 1, 5, 3)
    st.divider()
    st.caption(f"API: {API}")

tab_rec, tab_hist = st.tabs(["Recognise", "History"])

with tab_rec:
    left, right = st.columns(2)
    with left:
        up = st.file_uploader("Photo of a chessboard", type=["jpg", "jpeg", "png", "webp"])
        samples = sorted(SAMPLES.glob("*.jpg")) if SAMPLES.exists() else []
        pick = st.selectbox(
            "…or use a sample (ChessReD test split)",
            [None, *samples],
            format_func=lambda p: "—" if p is None else p.name,
        )
        data, name, ctype = None, None, "image/jpeg"
        if up is not None:
            data, name, ctype = up.getvalue(), up.name, up.type
        elif pick is not None:
            data, name = pick.read_bytes(), pick.name
        if data and st.button("Recognise", type="primary"):
            with st.spinner("Detecting board and pieces…"):
                form = {"turn": turn}
                if rot_choice != "auto":
                    form["rotation"] = str(rot_choice)
                res = api("POST", "/v1/recognize", files={"file": (name, data, ctype)}, data=form)
            if res:
                st.session_state["rec"] = res
                st.session_state["fen"] = res["fen"]
                st.session_state.pop("analysis", None)

    res = st.session_state.get("rec")
    if res:
        with left:
            show_b64(res["annotated_image"], "Detections and board outline")
            show_b64(res["board_image"], "Top-down view (white at bottom)")
            st.caption(
                f"corners: {res['corner_source']} ({res['corner_confidence']:.2f}) · "
                f"{res['timings_ms']['total']:.0f} ms · rotation {res['rotation']}"
            )
        with right:
            for w in res["warnings"]:
                st.warning(w)
            fen = st.text_input("FEN (edit to correct mistakes)", key="fen")
            err = fen_error(fen)
            if err:
                st.error(err)
            else:
                fen = (
                    set_turn(fen, turn) if st.checkbox("Apply side-to-move setting", True) else fen
                )
            best = (st.session_state.get("analysis") or {}).get("best_move")
            try:
                st.image(svg_data_uri(fen, best), width=440, caption="Board from the FEN above")
            except ValueError:
                pass
            c1, c2 = st.columns(2)
            if c1.button("Analyse with Stockfish", disabled=bool(err)):
                with st.spinner("Stockfish thinking…"):
                    a = api(
                        "POST", "/v1/analyze", json={"fen": fen, "depth": depth, "multipv": lines}
                    )
                if a:
                    st.session_state["analysis"] = a
                    st.rerun()
            c2.link_button("Open in Lichess", lichess_url(fen))
            a = st.session_state.get("analysis")
            if a:
                if a["game_over"]:
                    st.info("Game over in this position.")
                else:
                    st.success(
                        f"Best move **{a['best_move_san']}** ({a['best_move']}) · "
                        f"eval {eval_text(a['score_cp'], a['mate'])} (depth {a['depth']})"
                    )
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "move": ln["san"],
                                    "eval": eval_text(ln["score_cp"], ln["mate"]),
                                    "line": " ".join(ln["pv"]),
                                }
                                for ln in a["lines"]
                            ]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )

with tab_hist:
    h = api("GET", "/v1/history", params={"limit": 50})
    if h:
        st.write(f"{h['total']} recognitions stored")
        if h["items"]:
            df = pd.DataFrame(h["items"])[
                ["id", "created_at", "filename", "fen", "model_kind", "corner_source", "latency_ms"]
            ]
            st.dataframe(df, hide_index=True, use_container_width=True)
