"""Local, pretrained YOLO road-object detection."""

import os
from pathlib import Path
import time

import cv2

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo"))
MODEL_PATH = ROOT / "models" / "yolo11n.pt"
# COCO class IDs: person, bicycle, car, motorcycle, bus, train, truck,
# traffic light, stop sign, cat, dog, horse, sheep, cow.
ROAD_CLASSES = [0, 1, 2, 3, 5, 6, 7, 9, 11, 15, 16, 17, 18, 19]
_model = None
_status = {"count": 0, "labels": [], "fps": 0}


def detection_status():
    return dict(_status)


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise RuntimeError("Detection model missing. Run python setup_detection.py first, or turn off Detect road objects.")
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO
        _model = YOLO(str(MODEL_PATH))
    return _model


def annotate(frame, confidence=0.25):
    global _status
    model = get_model()
    started = time.perf_counter()
    result = model.predict(frame, conf=confidence, classes=ROAD_CLASSES,
                           imgsz=640, device="cpu", verbose=False, save=False)[0]
    annotated = result.plot(line_width=2, font_size=14)
    duration = time.perf_counter() - started
    count = len(result.boxes)
    labels = [f"{result.names[int(box.cls.item())]} {float(box.conf.item()):.0%}"
              for box in result.boxes]
    _status = {"count": count, "labels": labels, "fps": round(1 / max(duration, 0.001), 1)}
    text = f"ROAD DETECTION | {count} objects | {1 / max(duration, 0.001):.1f} inference FPS"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 34), (20, 32, 40), -1)
    cv2.putText(annotated, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (118, 232, 192), 1, cv2.LINE_AA)
    return annotated
