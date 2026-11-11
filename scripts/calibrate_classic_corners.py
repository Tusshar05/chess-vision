"""Measure the playing-area margin inside the detected mat quad on TRAIN-split images only."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ml.chessvision.board import order_corners
from ml.chessvision.corners_classic import WORK, _largest_quad
from ml.data.chessred import CORNER_ORDER, Annotations, canonical_corners

root = Path("data")
ann = Annotations(root / "raw" / "annotations.json")
man = json.loads((root / "local_manifest.json").read_text())
uvs = []
for iid in man["smoke_train"]:
    img = cv2.imread(str(root / "raw/images" / ann.images[iid]["file_name"]))
    s = WORK / max(img.shape[:2])
    gray = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
    quad = _largest_quad(gray)
    if quad is None:
        continue
    quad = order_corners(quad)
    H = cv2.getPerspectiveTransform(quad, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32))
    gt = np.array(canonical_corners([ann.corners[iid][k] for k in CORNER_ORDER]), np.float32) * s
    uv = cv2.perspectiveTransform(gt.reshape(-1, 1, 2), H).reshape(4, 2)
    if np.all(np.abs(uv - 0.5) < 0.5):  # sane
        uvs.append(uv)
print(len(uvs), "of", len(man["smoke_train"]), "train images gave a usable mat quad")
print(np.median(uvs, 0).round(3).tolist())
