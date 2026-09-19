# Local Cam

A Python webcam server with a local web viewer. Flask serves the page and OpenCV
captures the webcam connected to the computer running Python. No audio or video
is recorded. One viewer can stream at a time.

## Run on Windows

From this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_detection.py
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000 and click **Start camera**. Click **Stop** to release
the webcam. Press Ctrl+C in the terminal to shut down the server.

## Road-object detection

Leave **Detect road objects** checked to draw bounding boxes, class names, and
confidence scores on the live camera image. The first start loads the model and
may take several seconds. Stop the stream before changing camera or detection
settings. Lower the confidence threshold to show more possible detections;
raise it to reduce low-confidence detections. Leave Mirror preview off to keep
the labels readable.

The status below the preview reports the actual server detection results for
each displayed frame, including names, confidence scores, and the object count.
**Detection running · 0 object(s)** means the model ran but did not recognize a
supported object above the selected threshold. Try front lighting (avoid a bright
window behind the subject), a wider view of the person, or a lower threshold.

Click **Test detection with sample** while the camera is stopped to run the same
detector on a bundled street photograph. It should display boxes for a bus and
people. This isolates model problems from camera scene/lighting problems.
Run only one copy of `app.py`; after code changes stop it with Ctrl+C, restart it,
and refresh the browser. The web page rejects old streams without detection metadata.

The pipeline is: webcam → OpenCV frame → YOLO11 nano inference → annotated JPEG
→ Flask stream → web page. Inference runs locally on the CPU at a 640-pixel model
input size. Speed depends on your computer; the overlay shows inference FPS,
not total browser playback FPS. No camera images are sent to a cloud service.
Internet is needed to install dependencies and download weights once.

Supported classes: person, bicycle, car, motorcycle, bus, train, truck, traffic
light, stop sign, cat, dog, horse, sheep, and cow. This pretrained COCO model
does not detect potholes, lane markings, every road sign, or all possible hazards.
Traffic light detection identifies the object, not its red/amber/green state.
Boxes do not measure distance or speed. This is a demo, not a driving safety system.

For new classes, collect representative images, label the objects, train a custom
detector, and evaluate it on separate images before integrating it into this app.
The current detector and weights are from [Ultralytics](https://docs.ultralytics.com/modes/predict/);
see the provider's [license](https://www.ultralytics.com/license).

If detection fails, uncheck **Detect road objects** to use the original camera
stream. Run `.\.venv\Scripts\python.exe setup_detection.py` again if weights
are missing. Run `.\.venv\Scripts\python.exe -m unittest -v` for backend checks.

Choose camera 1 or 2 if your external webcam is not camera 0. If the camera fails
to open, close apps using it and enable camera access for desktop apps in Windows
Settings > Privacy & security > Camera. A disconnected camera requires restarting
the stream. Stopping or closing the page releases the camera after the server
detects the disconnected stream, usually within a few frames.

For another port: `.\.venv\Scripts\python.exe app.py --port 5050`.
The server binds to this computer only and runs with debug mode disabled.
On Windows, capture uses DirectShow explicitly. Run the server in a normal
PowerShell terminal; a restricted execution sandbox may block webcam access.

Implementation references: [Flask streaming](https://flask.palletsprojects.com/en/stable/patterns/streaming/)
and [OpenCV VideoCapture](https://docs.opencv.org/4.10.0/d8/dfe/classcv_1_1VideoCapture.html).
