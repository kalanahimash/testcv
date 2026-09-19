"""Download the official YOLO11 nano weights once, then verify inference."""

import urllib.request
import argparse

import numpy as np

from detector import MODEL_PATH, annotate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
        temporary = MODEL_PATH.with_suffix(".download")
        print("Downloading official YOLO11 nano model...", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            temporary.replace(MODEL_PATH)
        finally:
            temporary.unlink(missing_ok=True)
    if not args.download_only:
        annotate(np.zeros((480, 640, 3), dtype=np.uint8))
        print(f"Detection is ready: {MODEL_PATH}")
    else:
        print(f"Model downloaded: {MODEL_PATH}")
