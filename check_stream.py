"""Verify real MJPEG frames without saving webcam images."""

import argparse
import urllib.request

import cv2
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/stream?camera=0&detect={int(args.detect)}"
    with urllib.request.urlopen(url, timeout=90) as response:
        for index in range(3):
            line = response.readline()
            while line.strip() != b"--frame":
                if not line:
                    raise RuntimeError("Stream ended before a frame arrived")
                line = response.readline()
            headers = {}
            while (line := response.readline()).strip():
                key, value = line.decode().split(":", 1)
                headers[key.lower()] = value.strip()
            data = response.read(int(headers["content-length"]))
            if args.detect:
                if headers.get("x-detection") != "on":
                    raise RuntimeError("The server did not confirm detection; restart app.py")
                print("Detection result:", headers.get("x-detection-result"))
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("Stream returned an invalid JPEG")
            print(f"Frame {index + 1}: HTTP {response.status}, {frame.shape[1]}x{frame.shape[0]}, {len(data)} JPEG bytes")
