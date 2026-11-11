"""Data pipeline logic on a tiny synthetic annotations file (no downloads)."""

import json

import numpy as np
import pytest

from ml.data import chessred as C


@pytest.fixture()
def ann(tmp_path):
    images = [
        {
            "id": i,
            "file_name": f"G{g:03d}_IMG{i:03d}.jpg",
            "camera": "cam",
            "height": 1000,
            "width": 2000,
            "game_id": g,
        }
        for i, g in enumerate([0, 0, 1, 2])
    ]
    pieces = [
        {"image_id": 0, "category_id": 5, "chessboard_position": "e1", "bbox": [100, 200, 50, 100]},
        {"image_id": 0, "category_id": 11, "chessboard_position": "e8", "bbox": [300, 100, 40, 80]},
        {"image_id": 0, "category_id": 12, "chessboard_position": "a1"},
    ]
    corners = [
        {
            "image_id": 0,
            "corners": {
                "top_left": [1500, 900],
                "top_right": [1500, 100],  # board-relative labels, rotated image
                "bottom_right": [500, 100],
                "bottom_left": [500, 900],
            },
        }
    ]
    d = {
        "images": images,
        "annotations": {"pieces": pieces, "corners": corners},
        "categories": [{"id": i, "name": n} for i, n in enumerate(C.CLASS_NAMES + ["empty"])],
        "splits": {
            "train": {"image_ids": [0, 1]},
            "val": {"image_ids": [2]},
            "test": {"image_ids": [3]},
            "chessred2k": {k: {"image_ids": [0]} for k in ("train", "val", "test")},
        },
    }
    p = tmp_path / "a.json"
    p.write_text(json.dumps(d))
    return C.Annotations(p)


def test_no_game_leakage_passes(ann) -> None:
    games = ann.assert_no_game_leakage()
    assert games == {"train": [0], "val": [1], "test": [2]}


def test_game_leakage_detected(ann) -> None:
    ann.splits_full["test"] = [0]  # image from game 0 also in train
    with pytest.raises(AssertionError):
        ann.assert_no_game_leakage()


def test_fen_placement_from_annotations(ann) -> None:
    assert ann.fen_placement(0) == "4k3/8/8/8/8/8/8/4K3"


def test_yolo_detect_lines_normalised_and_skip_empty(ann) -> None:
    lines = C.yolo_detect_lines(ann, 0)
    assert len(lines) == 2
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "5" and float(cx) == pytest.approx((100 + 25) / 2000)
    assert float(cy) == pytest.approx((200 + 50) / 1000) and float(h) == pytest.approx(0.1)


def test_canonical_corners_starts_at_image_top_left() -> None:
    pts = [[1500, 900], [1500, 100], [500, 100], [500, 900]]  # clockwise, starts bottom-right-ish
    out = C.canonical_corners(pts)
    assert out[0] == [500, 100] and len(out) == 4


def test_canonical_corners_preserves_cyclic_order() -> None:
    pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
    for shift in range(4):
        rotated = pts[shift:] + pts[:shift]
        assert C.canonical_corners(rotated) == pts


def test_yolo_pose_line_has_four_visible_keypoints(ann) -> None:
    parts = C.yolo_pose_line(ann, 0).split()
    assert parts[0] == "0" and len(parts) == 5 + 12
    assert parts[7::3] == ["2"] * 4
    assert C.yolo_pose_line(ann, 1) is None


def test_md5_file(tmp_path) -> None:
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    assert C.md5_file(f) == "900150983cd24fb0d6963f7d28e17f72"


def test_write_yolo_dataset(ann, tmp_path) -> None:
    import cv2

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    cv2.imwrite(str(imgs / "G000_IMG000.jpg"), np.zeros((1000, 2000, 3), np.uint8))
    counts = C.write_yolo_dataset(ann, {"train": [0]}, imgs, tmp_path / "out")
    assert counts == {"train": 1}
    assert (tmp_path / "out/detect/labels/train/G000_IMG000.txt").read_text().count("\n") == 1
    import yaml

    y = yaml.safe_load((tmp_path / "out/pose.yaml").read_text())
    assert y["kpt_shape"] == [4, 3] and y["val"] == "images/train"  # aliased for Ultralytics
