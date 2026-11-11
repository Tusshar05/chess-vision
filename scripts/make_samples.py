"""Copy a few ChessReD2K *test-split* photos (resized to 1280 px) into samples/ with their FEN."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from ml.data.chessred import Annotations

root = Path("data")
ann = Annotations(root / "raw/annotations.json")
ids = json.loads((root / "local_manifest.json").read_text())["eval_test"]
by_cam: dict[str, list[int]] = {}
for i in ids:
    by_cam.setdefault(ann.images[i]["camera"], []).append(i)
out = Path("samples")
out.mkdir(exist_ok=True)
labels = {}
for cam, lst in sorted(by_cam.items()):
    for iid in (lst[len(lst) // 3], lst[2 * len(lst) // 3]):
        name = ann.images[iid]["file_name"]
        img = cv2.imread(str(root / "raw/images" / name))
        img = cv2.resize(img, (1280, 1280), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        labels[name] = {"camera": cam, "fen_placement": ann.fen_placement(iid), "image_id": iid}
(out / "labels.json").write_text(json.dumps(labels, indent=2))
(out / "README.md").write_text(
    "# Sample photos\n\nSix photos from the **test split** (ChessReD2K) of the Chess Recognition "
    "Dataset (ChessReD), resized to 1280 px.\n\n"
    "- Source: Masouris & van Gemert, *End-to-End Chess Recognition*, VISAPP 2024. "
    "4TU.ResearchData, DOI 10.4121/99b5c721-280b-450b-b058-b2900b69a90f.v2\n"
    "- License: **CC BY-NC-SA 4.0** (as listed by 4TU.ResearchData) - attribution, non-commercial, "
    "share-alike. These resized copies are shared under the same license.\n"
    "- `labels.json`: ground-truth piece placement (FEN field 1) per photo.\n"
)
print(len(labels), "samples")
