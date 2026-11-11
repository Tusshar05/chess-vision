"""Bootstrap confidence intervals for board-level metrics."""

from __future__ import annotations

import numpy as np

METRICS = {
    "boards_0_errors_pct": lambda e: 100 * float((e == 0).mean()),
    "boards_le1_errors_pct": lambda e: 100 * float((e <= 1).mean()),
    "mean_errors_per_board": lambda e: float(e.mean()),
}


def bootstrap_ci(
    errors: list[int], games: list[int] | None = None, n: int = 2000, seed: int = 0
) -> dict:
    """95% percentile CIs for the board metrics.

    ``board``: resample boards independently (assumes independent boards - optimistic because
    photos of one game are correlated). ``game``: resample whole games (cluster bootstrap), the
    honest version when there are few games; with very few games it is very wide.
    """
    e = np.asarray(errors)
    rng = np.random.default_rng(seed)
    out: dict = {"n_boards": len(e)}
    idx = rng.integers(0, len(e), size=(n, len(e)))
    out["board_bootstrap"] = {
        k: [round(float(np.percentile([f(e[i]) for i in idx], q)), 2) for q in (2.5, 97.5)]
        for k, f in METRICS.items()
    }
    if games is not None:
        g = np.asarray(games)
        uniq = np.unique(g)
        groups = [e[g == u] for u in uniq]
        draws = rng.integers(0, len(uniq), size=(n, len(uniq)))
        samples = [np.concatenate([groups[j] for j in d]) for d in draws]
        out["n_games"] = len(uniq)
        out["game_cluster_bootstrap"] = {
            k: [round(float(np.percentile([f(s) for s in samples], q)), 2) for q in (2.5, 97.5)]
            for k, f in METRICS.items()
        }
    return out
