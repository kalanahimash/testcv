"""Local webcam viewer. Run with: python app.py"""

import argparse
import math
import json
import sys
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template, request
from detector import annotate, detection_status

app = Flask(__name__)
camera_lock = threading.Lock()


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/detection-demo")
def detection_demo():
    if not camera_lock.acquire(blocking=False):
        return jsonify(error="Stop the camera before running the sample test."), 409
    try:
        sign_demo = request.args.get("signs", "0") == "1"
        if sign_demo:
            from sign_samples import sample_board
            sample = sample_board()
        else:
            from detector import get_model
            get_model()
            from ultralytics.utils import ASSETS
            sample = cv2.imread(str(ASSETS / "bus.jpg"))
        if sample is None:
            raise RuntimeError("Bundled sample image is missing")
        annotated = annotate(sample, 0.25, signs=sign_demo, objects=not sign_demo)
        ok, jpeg = cv2.imencode(".jpg", annotated)
        if not ok:
            raise RuntimeError("Could not encode sample image")
        return Response(jpeg.tobytes(), mimetype="image/jpeg", headers={
            "X-Detection": "on", "X-Detection-Result": json.dumps(detection_status())})
    except Exception as error:
        app.logger.exception("Sample detection failed")
        return jsonify(error=f"Sample detection failed: {error}"), 503
    finally:
        camera_lock.release()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/sign-test-card")
def sign_test_card():
    from sign_samples import sample_board
    ok, jpeg = cv2.imencode(".jpg", sample_board())
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


@app.get("/stream")
def stream():
    try:
        camera_index = int(request.args.get("camera", "0"))
        if not 0 <= camera_index <= 10:
            raise ValueError
    except ValueError:
        return jsonify(error="Camera number must be between 0 and 10."), 400

    detect = request.args.get("detect", "0") == "1"
    signs = request.args.get("signs", "0") == "1"
    processing = detect or signs
    try:
        confidence = float(request.args.get("confidence", "0.25"))
        if not math.isfinite(confidence) or not 0.1 <= confidence <= 0.9:
            raise ValueError
    except ValueError:
        return jsonify(error="Confidence must be between 0.1 and 0.9."), 400

    if not camera_lock.acquire(blocking=False):
        return jsonify(error="Camera is already streaming. Stop the other viewer first."), 409

    capture = None
    released = False

    def cleanup():
        nonlocal released
        if not released:
            released = True
            if capture is not None:
                capture.release()
            camera_lock.release()

    try:
        # Try multiple backends (DirectShow, MSMF, and ANY) for robust USB webcam support on Windows
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        for b in backends:
            capture = cv2.VideoCapture(camera_index, b)
            if capture.isOpened():
                break
        if not capture.isOpened():
            cleanup()
            return jsonify(error=f"Cannot open camera {camera_index}. If you plugged in a USB webcam or disabled your built-in webcam, please select '1 · USB / External camera' from the Camera dropdown."), 503
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, first_frame = capture.read()
        if not ok:
            cleanup()
            return jsonify(error=f"Camera {camera_index} opened but did not provide an image. Try another camera index in the dropdown."), 503
    except Exception:
        cleanup()
        app.logger.exception("Camera initialization failed")
        return jsonify(error="Camera initialization failed. See the Python terminal for details."), 503

    if processing:
        try:
            first_frame = annotate(first_frame, confidence, signs=signs, objects=detect)
        except Exception as error:
            cleanup()
            app.logger.exception("Detection initialization failed")
            return jsonify(error=f"Detection could not start: {error}. You can turn off Detect road objects to use the camera alone."), 503

    def frames():
        frame = first_frame
        try:
            while True:
                started = time.monotonic()
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    break
                data = encoded.tobytes()
                metadata = ("X-Detection: " + ("on" if processing else "off") + "\r\n"
                            + "X-Detection-Result: " + json.dumps(detection_status() if processing else {})
                            + "\r\n").encode("ascii")
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(data)).encode() + b"\r\n" + metadata + b"\r\n" + data + b"\r\n")
                if not processing:
                    time.sleep(max(0, 1 / 30 - (time.monotonic() - started)))
                ok, frame = capture.read()
                if not ok:
                    break
                if processing:
                    frame = annotate(frame, confidence, signs=signs, objects=detect)
        except Exception:
            app.logger.exception("Camera stream or detection failed")
        finally:
            cleanup()

    response = Response(frames(), content_type="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})
    response.call_on_close(cleanup)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)
