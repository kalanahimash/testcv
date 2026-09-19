"""Download the Sri Lankan traffic-sign research model and verify its source hash."""

import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "models" / "sri_lanka_signs.pt"
SOURCE = "https://raw.githubusercontent.com/supun-chamika/Sri-Lankan-Traffic-Sign-Detection-with-YOLOv8/main/GUI/trained-models/best.pt"
EXPECTED_GIT_BLOB = "52ec3d8ce7b24e1cf08ca37b570d33292166ff97"


def verify(data):
    digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if digest != EXPECTED_GIT_BLOB:
        raise RuntimeError("Downloaded model differs from the inspected source. Refusing to use it.")


if __name__ == "__main__":
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists():
        verify(DESTINATION.read_bytes())
    else:
        print("Downloading Sri Lankan traffic-sign model...", flush=True)
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            data = response.read()
        verify(data)
        DESTINATION.write_bytes(data)
    print(f"Verified model: {DESTINATION}")
