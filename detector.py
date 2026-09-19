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


def annotate(frame, confidence=0.25, signs=False, objects=True):
    global _status
    started = time.perf_counter()
    labels = []
    annotated = frame.copy()
    sign_results = []
    if signs:
        from sign_detector import detect_signs, draw_signs
        sign_results = detect_signs(frame, max(confidence, 0.35))
        draw_signs(annotated, sign_results)
        labels.extend(f'{item["name"]} {item["confidence"]:.0%}' for item in sign_results)

    if objects:
        model = get_model()
        result = model.predict(frame, conf=confidence, classes=ROAD_CLASSES,
                               imgsz=640, device="cpu", verbose=False, save=False)[0]
        from sign_detector import is_actual_stop_sign, box_iou
        speed_boxes = [s["box"] for s in sign_results]
        
        for box in result.boxes:
            cls_id = int(box.cls.item())
            cls_name = result.names[cls_id]
            conf_val = float(box.conf.item())
            b = [int(n) for n in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = b
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

            if cls_name == "stop sign":
                # Suppress if overlaps with any detected road/speed sign or if interior is not solid red
                if any(box_iou(b, sbox) > 0.20 for sbox in speed_boxes):
                    continue
                if not is_actual_stop_sign(crop):
                    continue

            # Draw road object bounding box
            color = (80, 220, 100) if cls_name == "person" else (220, 140, 60)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf_val:.0%}"
            (w_t, h_t), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(h_t + 4, y1 - 4)
            tx = max(0, min(x1, annotated.shape[1] - w_t - 4))
            cv2.rectangle(annotated, (tx, ty - h_t - 2), (tx + w_t + 4, ty + baseline), color, -1)
            cv2.putText(annotated, label, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 15, 20), 1, cv2.LINE_AA)
            labels.append(label)

    duration = time.perf_counter() - started
    count = len(labels)
    _status = {"count": count, "labels": labels, "fps": round(1 / max(duration, 0.001), 1),
               "signs_enabled": signs, "objects_enabled": objects,
               "signs": sign_results, "speed_limits": [s["speed"] for s in sign_results if "speed" in s]}
    text = f"ROAD DETECTION | {count} objects | {1 / max(duration, 0.001):.1f} inference FPS"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 34), (20, 32, 40), -1)
    cv2.putText(annotated, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (118, 232, 192), 1, cv2.LINE_AA)
    return annotated
