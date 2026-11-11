"""ChessReD dataset access: annotations, official splits, remote-zip fetching, YOLO conversion.

Dataset: Masouris & van Gemert, "End-to-End Chess Recognition" (VISAPP 2024),
4TU.ResearchData, DOI 10.4121/99b5c721-280b-450b-b058-b2900b69a90f.v2
(license as listed by 4TU: CC BY-NC-SA 4.0).
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests

DOI = "10.4121/99b5c721-280b-450b-b058-b2900b69a90f.v2"
BASE = "https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f"


@dataclass(frozen=True)
class RemoteFile:
    """A downloadable file of the dataset with its published size and MD5."""

    name: str
    url: str
    size: int
    md5: str


# Values read from the 4TU.ResearchData API (data.4tu.nl/v2/articles/<uuid>) on 2026-11-11.
ANNOTATIONS = RemoteFile(
    "annotations.json",
    f"{BASE}/3cae6364-daca-4967-b426-1e4b68cdb64c",
    22204010,
    "d34bca5ad46ec7a8df96a1d3c36784f3",
)
CHESSRED2K = RemoteFile(
    "chessred2k.zip",
    f"{BASE}/410e41c7-dde5-413f-8722-3e112363a1a2",
    4680150747,
    "4e55e299b6b2973e0d8659bd241c7975",
)
CHESSRED_FULL = RemoteFile(
    "chessred.zip",
    f"{BASE}/6329e969-616e-48e3-b893-a0379d1c15ba",
    24573697524,
    "32e23ed535239dc517a6499762d9847e",
)

# Category ids 0..11 as in annotations.json (12 = "empty", not a detectable object).
CLASS_NAMES = [
    "white-pawn",
    "white-rook",
    "white-knight",
    "white-bishop",
    "white-queen",
    "white-king",
    "black-pawn",
    "black-rook",
    "black-knight",
    "black-bishop",
    "black-queen",
    "black-king",
]
CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]
MAX_SIDE = 1280  # images are resized on conversion; YOLO labels are normalised so unaffected


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(rf: RemoteFile, dest_dir: Path, verify: bool = True) -> Path:
    """Download ``rf`` into ``dest_dir`` with resume support and checksum verification."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / rf.name
    if out.exists() and out.stat().st_size == rf.size:
        if not verify or md5_file(out) == rf.md5:
            return out
        out.unlink()
    part = out.with_suffix(out.suffix + ".part")
    pos = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(rf.url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        mode = "ab" if pos and r.status_code == 206 else "wb"
        with open(part, mode) as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    part.rename(out)
    if verify and md5_file(out) != rf.md5:
        raise RuntimeError(f"MD5 mismatch for {rf.name}; delete {out} and retry")
    return out


class Annotations:
    """Parsed ``annotations.json`` with convenience lookups."""

    def __init__(self, path: Path):
        d = json.loads(Path(path).read_text())
        self.images: dict[int, dict] = {i["id"]: i for i in d["images"]}
        self.pieces: dict[int, list[dict]] = defaultdict(list)
        for p in d["annotations"]["pieces"]:
            self.pieces[p["image_id"]].append(p)
        self.corners: dict[int, dict] = {
            c["image_id"]: c["corners"] for c in d["annotations"]["corners"]
        }
        self.splits_full = {k: d["splits"][k]["image_ids"] for k in ("train", "val", "test")}
        self.splits_2k = {k: d["splits"]["chessred2k"][k]["image_ids"] for k in ("train", "val", "test")}
        self.categories = {c["id"]: c["name"] for c in d["categories"]}

    def game_of(self, image_id: int) -> int:
        """Game id of an image."""
        return self.images[image_id]["game_id"]

    def assert_no_game_leakage(self) -> dict[str, list[int]]:
        """Verify splits are disjoint by game (the official split is); return games per split."""
        games = {
            s: sorted({self.game_of(i) for i in ids}) for s, ids in self.splits_full.items()
        }
        names = list(games)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                overlap = set(games[names[a]]) & set(games[names[b]])
                if overlap:
                    raise AssertionError(f"games in both {names[a]} and {names[b]}: {overlap}")
        return games

    def fen_placement(self, image_id: int) -> str:
        """Ground-truth FEN piece-placement field from the per-square piece annotations."""
        from ml.chessvision.constants import CLASS_TO_SYMBOL

        grid = {}
        for p in self.pieces[image_id]:
            if p["category_id"] < 12:
                grid[p["chessboard_position"]] = CLASS_TO_SYMBOL[p["category_id"]]
        rows = []
        for rank in range(8, 0, -1):
            row, empty = "", 0
            for f in "abcdefgh":
                sym = grid.get(f"{f}{rank}")
                if sym is None:
                    empty += 1
                else:
                    row += (str(empty) if empty else "") + sym
                    empty = 0
            rows.append(row + (str(empty) if empty else ""))
        return "/".join(rows)


class _HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (lets ``zipfile`` read remote zips)."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0
        self.session = requests.Session()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        r = self.session.get(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}, timeout=120
        )
        r.raise_for_status()
        self.pos += len(r.content)
        return r.content


def fetch_from_remote_zip(rf: RemoteFile, wanted_basenames: list[str], out_dir: Path,
                          workers: int = 8) -> list[Path]:
    """Download only the named image files out of a remote zip via HTTP Range requests.

    Avoids pulling the multi-GB archive when only a sample/eval subset is needed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [b for b in wanted_basenames if not (out_dir / b).exists()]
    if todo:
        zf = zipfile.ZipFile(_HTTPRangeFile(rf.url, rf.size))
        index = {Path(i.filename).name: i for i in zf.infolist() if not i.is_dir()}
        missing = [b for b in todo if b not in index]
        if missing:
            raise FileNotFoundError(f"{len(missing)} files not in zip, e.g. {missing[:3]}")

        def one(base: str) -> None:
            local = zipfile.ZipFile(_HTTPRangeFile(rf.url, rf.size))
            (out_dir / base).write_bytes(local.read(index[base].filename))

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
    return [out_dir / b for b in wanted_basenames]


def yolo_detect_lines(ann: Annotations, image_id: int) -> list[str]:
    """YOLO detection label lines (class cx cy w h, normalised) for one image."""
    im = ann.images[image_id]
    W, H = im["width"], im["height"]
    lines = []
    for p in ann.pieces[image_id]:
        if "bbox" not in p or p["category_id"] >= 12:
            continue
        x, y, w, h = p["bbox"]
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        lines.append(f"{p['category_id']} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
    return lines


def canonical_corners(corners: list[list[float]]) -> list[list[float]]:
    """Rotate a clockwise corner list so it starts at the image-space top-left corner.

    ChessReD labels corners by *board* identity (``top_left`` is always the a8 corner), so on
    photos taken from the side they are rotated in image space. The corner model is trained on
    this image-space canonical order (TL, TR, BR, BL); board orientation is resolved later
    from the piece distribution (see ``ml.chessvision.board.resolve_orientation``).
    """
    start = min(range(4), key=lambda i: corners[i][0] + corners[i][1])
    return [list(corners[(start + k) % 4]) for k in range(4)]


def yolo_pose_line(ann: Annotations, image_id: int, pad: float = 0.02) -> str | None:
    """YOLO-pose label (single class 'board', 4 keypoints TL,TR,BR,BL) or None if no corners."""
    c = ann.corners.get(image_id)
    if c is None:
        return None
    im = ann.images[image_id]
    W, H = im["width"], im["height"]
    pts = canonical_corners([c[k] for k in CORNER_ORDER])
    pts = [(x / W, y / H) for x, y in pts]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1 = max(0.0, min(xs) - pad), min(1.0, max(xs) + pad)
    y0, y1 = max(0.0, min(ys) - pad), min(1.0, max(ys) + pad)
    kp = " ".join(f"{min(max(x, 0), 1):.6f} {min(max(y, 0), 1):.6f} 2" for x, y in pts)
    return f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f} {kp}"


def write_yolo_dataset(ann: Annotations, split_ids: dict[str, list[int]], image_dir: Path,
                       out_root: Path, limit: dict[str, int] | None = None) -> dict[str, int]:
    """Write detector + corner-pose YOLO datasets (resized images, labels, data yamls).

    ``image_dir`` holds the original ``G###_IMG###.jpg`` files. Images are downscaled to
    ``MAX_SIDE`` and symlinked-by-copy into both the detect/ and pose/ trees.
    """
    import cv2
    import yaml

    counts: dict[str, int] = {}
    for split, ids in split_ids.items():
        if limit and split in limit:
            ids = ids[:: max(1, len(ids) // limit[split])][: limit[split]]
        n = 0
        for task in ("detect", "pose"):
            (out_root / task / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_root / task / "labels" / split).mkdir(parents=True, exist_ok=True)
        for iid in ids:
            name = ann.images[iid]["file_name"]
            src = image_dir / name
            if not src.exists():
                continue
            img = cv2.imread(str(src))
            if img is None:
                continue
            s = MAX_SIDE / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            stem = Path(name).stem
            for task, lines in (
                ("detect", yolo_detect_lines(ann, iid)),
                ("pose", [ln] if (ln := yolo_pose_line(ann, iid)) else []),
            ):
                cv2.imwrite(str(out_root / task / "images" / split / f"{stem}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out_root / task / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
            n += 1
        counts[split] = n
    names = {i: n for i, n in enumerate(CLASS_NAMES)}
    base = {"path": str(out_root.resolve())}
    splits_present = {s: f"images/{s}" for s in counts if counts[s]}
    for required in ("train", "val"):  # Ultralytics insists on both; alias for test-only trees
        splits_present.setdefault(required, next(iter(splits_present.values())))
    (out_root / "detect.yaml").write_text(
        yaml.safe_dump({**base, "path": str((out_root / "detect").resolve()), **splits_present,
                        "names": names}))
    (out_root / "pose.yaml").write_text(
        yaml.safe_dump({"path": str((out_root / "pose").resolve()), **splits_present,
                        "kpt_shape": [4, 3], "flip_idx": [1, 0, 3, 2], "names": {0: "board"}}))
    return counts
