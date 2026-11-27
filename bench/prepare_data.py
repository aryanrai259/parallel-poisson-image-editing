#!/usr/bin/env python3
"""Create the benchmark inputs in bench/data.

squareX / circleX: synthetic masks with about 4**X unknowns (same generator as
tests/data.py). testN_*: the photo examples listed in tests/data.txt.
"""

import argparse
import os
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LINKS = HERE.parent / "tests" / "data.txt"


def square(x: int) -> None:
    r = int((4**x) ** 0.5 + 2)
    img = np.zeros([r, r, 3], np.uint8) + 255
    cv2.imwrite(str(DATA / f"square{x}.png"), img)


def circle(x: int) -> None:
    r = int(((4**x) * 4 / np.pi) ** 0.5 + 2)
    img = np.zeros([r, r, 3], np.uint8)
    img = cv2.circle(img, (r // 2, r // 2), r // 2, (255, 255, 255), -1)
    cv2.imwrite(str(DATA / f"circle{x}.png"), img)


def download() -> None:
    for line in LINKS.read_text().splitlines():
        if not line.strip():
            continue
        url, name = line.split()
        dst = DATA / name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
            print(f"downloaded {name}")
        except Exception as exc:  # network failures leave the example out
            if dst.exists():
                dst.unlink()
            print(f"skipped {name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    os.makedirs(DATA, exist_ok=True)
    for x in range(6, 11):
        square(x)
        circle(x)
    if not args.no_download:
        download()
    print(f"data in {DATA}")


if __name__ == "__main__":
    main()
