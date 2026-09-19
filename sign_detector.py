"""Sri Lankan roadside signs plus speed-sign OCR with multi-line support."""

import importlib
import re

import cv2
import numpy as np

from detector import ROOT
from setup_signs import DESTINATION, verify

_sign_model = None
_ocr = None

SAFE_CLASSES = (
    "ultralytics.nn.modules.conv.Conv", "ultralytics.nn.modules.block.Bottleneck",
    "ultralytics.nn.modules.block.C2f", "ultralytics.nn.modules.block.SPPF",
    "ultralytics.utils.IterableSimpleNamespace", "torch.nn.modules.activation.SiLU",
    "torch.nn.modules.batchnorm.BatchNorm2d", "ultralytics.utils.tal.TaskAlignedAssigner",
    "torch.nn.modules.upsampling.Upsample", "torch.nn.modules.pooling.MaxPool2d",
    "ultralytics.utils.loss.BboxLoss", "torch.nn.modules.container.Sequential",
    "ultralytics.nn.modules.conv.Concat", "torch.nn.modules.conv.Conv2d",
    "torch.nn.modules.container.ModuleList", "ultralytics.utils.loss.v8DetectionLoss",
    "ultralytics.nn.modules.block.DFL", "torch.nn.modules.loss.BCEWithLogitsLoss",
    "ultralytics.nn.tasks.DetectionModel", "ultralytics.nn.modules.head.Detect",
)

VALID_SPEED_MIN = 5
VALID_SPEED_MAX = 160

DIGIT_MAP = {
    'O': '0', 'o': '0', 'Q': '0', 'D': '0',
    'I': '1', 'l': '1', '|': '1', '!': '1', 'i': '1',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5',
    'B': '8',
}


def get_sign_model():
    global _sign_model
    if _sign_model is None:
        if not DESTINATION.is_file():
            raise RuntimeError("Run python setup_signs.py to download the Sri Lankan sign model")
        verify(DESTINATION.read_bytes())
        (ROOT / ".yolo").mkdir(exist_ok=True)
        import torch
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
        allowed = [getattr(importlib.import_module(name.rsplit('.', 1)[0]), name.rsplit('.', 1)[1])
                   for name in SAFE_CLASSES]
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(DESTINATION, map_location="cpu", weights_only=True)
        network = (checkpoint.get("ema") or checkpoint["model"]).float().eval()
        network.args = dict(DEFAULT_CFG_DICT)
        network.task = "detect"
        network.pt_path = str(DESTINATION)
        wrapper = YOLO("yolov8s.yaml", task="detect")
        wrapper.model = network
        wrapper.overrides = {"task": "detect", "imgsz": 640, "model": str(DESTINATION)}
        _sign_model = wrapper
    return _sign_model


def get_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)
    return _ocr


def is_actual_stop_sign(crop):
    """Verify if a detected sign is a genuine solid-red Stop sign rather than a circular speed sign."""
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    inner = crop[int(h * 0.20):int(h * 0.80), int(w * 0.20):int(w * 0.80)]
    if inner.size == 0:
        return False
    b, g, r = cv2.split(inner)
    red_pixels = (r.astype(int) - g.astype(int) > 25) & (r.astype(int) - b.astype(int) > 25) & (r > 60)
    # Genuine Stop signs have a solid red interior (>35% red pixels in the center)
    return float(red_pixels.mean()) >= 0.35


def speed_candidates(frame):
    """Find red circular borders with light interiors across various lighting conditions."""
    if frame is None or frame.size == 0:
        return []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Adaptive red mask: combination of HSV and RGB color difference
    mask1 = cv2.inRange(hsv, (0, 45, 40), (16, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 45, 40), (180, 255, 255))
    b, g, r = cv2.split(frame)
    rgb_red = (r.astype(int) - g.astype(int) > 20) & (r.astype(int) - b.astype(int) > 20) & (r > 50)
    red = mask1 | mask2 | (rgb_red.astype(np.uint8) * 255)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 28 or area < 450:
            continue
        aspect = w / float(h)
        if not (0.65 < aspect < 1.55):
            continue
        circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
        if circularity < 0.42:
            continue
        # Check center region (must have a light/non-red interior, not solid red)
        crop = frame[max(0, y):min(frame.shape[0], y + h), max(0, x):min(frame.shape[1], x + w)]
        if crop.size == 0:
            continue
        if is_actual_stop_sign(crop):
            continue
        candidates.append(((x, y, x + w, y + h), crop))
        if len(candidates) == 5:
            break
    return candidates


def parse_speed_text(text):
    """Extract any valid road speed limit (5 to 160 km/h) from OCR text with character correction."""
    if not text:
        return None
    cleaned = str(text).strip()
    
    # 1. Clean standalone numbers (e.g. '5', '50', '80', '100')
    matches = re.findall(r"\b(\d{1,3})\b", cleaned)
    for m in matches:
        speed = int(m)
        if VALID_SPEED_MIN <= speed <= VALID_SPEED_MAX:
            return speed

    # 2. Number attached to unit (e.g. '50kmph', '80km/h', '100kph')
    unit_match = re.search(r"(?i)\b(\d{1,3})\s*(?:km/?h|kmph|kph|mph)\b", cleaned)
    if unit_match:
        speed = int(unit_match.group(1))
        if VALID_SPEED_MIN <= speed <= VALID_SPEED_MAX:
            return speed

    # 3. Letter-digit substitutions (e.g. '8O' -> 80, '5O kmph' -> 50, '1OO' -> 100)
    text_no_unit = re.sub(r"(?i)\b(km/?h|kmph|kph|mph|speed|limit)\b", "", cleaned).strip()
    normalized = ""
    for char in text_no_unit:
        if char.isdigit():
            normalized += char
        elif char in DIGIT_MAP:
            normalized += DIGIT_MAP[char]
    if normalized:
        for m in re.findall(r"(\d{1,3})", normalized):
            speed = int(m)
            if VALID_SPEED_MIN <= speed <= VALID_SPEED_MAX:
                return speed

    # 4. Fallback raw digits
    for m in re.findall(r"(\d{1,3})", cleaned):
        speed = int(m)
        if VALID_SPEED_MIN <= speed <= VALID_SPEED_MAX:
            return speed
    return None


def detect_speed_signs(frame):
    """Detect speed signs and extract the speed limit number across all road speeds."""
    detections = []
    ocr_engine = get_ocr()
    for box, crop in speed_candidates(frame):
        # Resize crop if small for clearer OCR
        h, w = crop.shape[:2]
        target_crop = crop
        if max(h, w) < 220:
            scale = 220.0 / max(h, w)
            target_crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # 1. First attempt: Full OCR with text detection on target crop
        ocr_result, _ = ocr_engine(target_crop, use_det=True, use_cls=False, use_rec=True)
        
        # 2. Fallback attempt: OCR on inner 75% circle to eliminate any red-border border artifacts
        if not ocr_result:
            ih, iw = target_crop.shape[:2]
            inner_crop = target_crop[int(ih * 0.12):int(ih * 0.88), int(iw * 0.12):int(iw * 0.88)]
            if inner_crop.size > 0:
                ocr_result, _ = ocr_engine(inner_crop, use_det=True, use_cls=False, use_rec=True)

        # 3. Fallback attempt: direct recognition on tight digit bounding box
        if not ocr_result:
            gray = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY)
            _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            points = cv2.findNonZero(ink)
            if points is not None:
                dx, dy, dw, dh = cv2.boundingRect(points)
                if dh >= target_crop.shape[0] * 0.12:
                    digit_crop = target_crop[dy:dy+dh, dx:dx+dw]
                    digit_crop = cv2.copyMakeBorder(digit_crop, 8, 8, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    ocr_result, _ = ocr_engine(digit_crop, use_det=False, use_cls=False, use_rec=True)

        if not ocr_result:
            continue

        best_speed = None
        best_score = 0.0
        for item in ocr_result:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                txt = item[1]
                score = float(item[2]) if len(item) > 2 else 0.80
            else:
                txt = str(item)
                score = 0.80
            speed = parse_speed_text(txt)
            if speed is not None and score > best_score:
                best_speed = speed
                best_score = score

        if best_speed is not None and best_score >= 0.65:
            detections.append({
                "name": f"Speed limit {best_speed} km/h (OCR)",
                "confidence": float(best_score),
                "box": list(box),
                "speed": best_speed,
                "kind": "speed"
            })
    return detections


def box_iou(box_a, box_b):
    """Compute Intersection over Union between two boxes [x1, y1, x2, y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union_area = float(area_a + area_b - inter_area)
    return inter_area / union_area if union_area > 0 else 0.0


def detect_signs(frame, confidence=0.4):
    """Detect Sri Lankan road signs and speed limits, suppressing false Stop signs."""
    prediction = get_sign_model().predict(frame, conf=confidence, imgsz=640,
                                          device="cpu", verbose=False, save=False)[0]
    speed_detections = detect_speed_signs(frame)
    speed_boxes = [d["box"] for d in speed_detections]

    detections = []
    for box in prediction.boxes:
        name = prediction.names[int(box.cls.item())].replace('-', ' ')
        bounds = [int(n) for n in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = bounds
        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

        # If this detection overlaps with a recognized speed sign, suppress it
        if any(box_iou(bounds, sbox) > 0.25 for sbox in speed_boxes):
            continue

        # Suppress false Stop/Stop-Ahead signs on non-red objects or speed signs
        if "Stop" in name:
            if not is_actual_stop_sign(crop):
                continue

        detections.append({"name": name, "confidence": float(box.conf.item()),
                           "box": bounds, "kind": "sign"})

    return detections + speed_detections


def draw_signs(frame, detections):
    """Draw bounding boxes and badges for signs on the camera frame."""
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        color = (0, 215, 255) if item["kind"] == "speed" else (255, 170, 70)
        label = f'{item["name"]} {item["confidence"]:.0%}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .55, 2)
        text_y = max(height + 5, y1 - 8)
        text_x = max(0, min(x1, frame.shape[1] - width - 6))
        cv2.rectangle(frame, (text_x, text_y - height - 4), (text_x + width + 5, text_y + baseline), color, -1)
        cv2.putText(frame, label, (text_x + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, .55, (10, 15, 20), 2, cv2.LINE_AA)
    return frame

