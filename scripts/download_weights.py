"""Download the trained-weights bundle from a GitHub Release into release_assets/.

    WEIGHTS_URL=<release asset url> python scripts/download_weights.py [--sha256 <hex>]

Then run ``make weights``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import requests

DEFAULT_URL = (
    "https://github.com/Tusshar05/chess-vision/releases/download/v0.1.0/chessvision_weights.zip"
)


def main() -> int:
    """Fetch the bundle (streamed), optionally verify its checksum."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.environ.get("WEIGHTS_URL", DEFAULT_URL))
    ap.add_argument("--sha256", default=os.environ.get("WEIGHTS_SHA256"))
    ap.add_argument("--out", type=Path, default=Path("release_assets"))
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)
    dest = a.out / Path(a.url).name
    print("downloading", a.url)
    h = hashlib.sha256()
    with requests.get(a.url, stream=True, timeout=60) as r:
        if r.status_code != 200:
            print(f"download failed: HTTP {r.status_code} (has the release been published?)")
            return 1
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                h.update(chunk)
    print("saved", dest, h.hexdigest())
    if a.sha256 and a.sha256 != h.hexdigest():
        print("SHA-256 mismatch!")
        dest.unlink()
        return 1
    print("now run: make weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
