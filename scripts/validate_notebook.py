"""Dry-run the notebook's code cells locally (CHESSVISION_DRY_RUN=1) to catch errors early."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CHESSVISION_DRY_RUN"] = "1"
nb = json.loads(Path("notebooks/train_colab.ipynb").read_text())
ns: dict = {"__name__": "__main__"}
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    print(f"--- cell {i} ---")
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED cell {i}: {type(e).__name__}: {e}")
        sys.exit(1)
print("notebook dry-run OK")
