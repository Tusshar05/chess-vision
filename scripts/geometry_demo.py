"""Figure for the README: the geometry stage driven by *ground-truth* boxes and corners.

Left: photo with the ground-truth outline and boxes. Right: homography-warped top-down board with
squares assigned by ``reconstruct``. Shows the position-reconstruction logic, not a model result.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ml.chessvision import render
from ml.chessvision.board import Detection, reconstruct
from ml.data.chessred import CORNER_ORDER, Annotations

root = Path("data")
ann = Annotations(root / "raw/annotations.json")
ids = json.loads((root / "local_manifest.json").read_text())["eval_test"]
panels = []
for iid in (ids[40], ids[150], ids[260]):
    img = cv2.imread(str(root / "raw/images" / ann.images[iid]["file_name"]))
    c = np.array([ann.corners[iid][k] for k in CORNER_ORDER], np.float32)
    rot = min(range(4), key=lambda i: c[i].sum())
    dets = [
        Detection(
            p["bbox"][0],
            p["bbox"][1],
            p["bbox"][0] + p["bbox"][2],
            p["bbox"][1] + p["bbox"][3],
            p["category_id"],
            1.0,
        )
        for p in ann.pieces[iid]
        if "bbox" in p and p["category_id"] < 12
    ]
    rec = reconstruct(dets, c, rotation=rot)
    left = cv2.resize(render.annotate(img, c, dets), (520, 520))
    right = render.top_down(img, c, rec, 520)
    cv2.putText(
        left,
        "ground-truth boxes + corners",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        right, rec.fen_placement()[:34], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
    )
    panels.append(np.hstack([left, right]))
out = Path("reports/data/geometry_demo.jpg")
cv2.imwrite(str(out), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 80])
print("wrote", out)
