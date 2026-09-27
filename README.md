# FastTrack: Airplane Segmentation and Tracking

FastTrack is a Python desktop application for segmenting, localizing, and tracking airplanes in video. It uses a PyQt6 interface, OpenCV for video input/output, and Ultralytics YOLO segmentation weights for per-frame masks and detections.

The app overlays the airplane mask, bounding box, center point, track ID, and trail on the video. It also exports a CSV containing frame-by-frame localization and motion measurements.

Save README.md in the repository root beside fasttrack_pyqt.py so GitHub displays it on the project page.

## Features

- Instance segmentation masks for detected aircraft.
- Bounding boxes, center coordinates, confidence scores, track IDs, and motion trails.
- ByteTrack, BoT-SORT, and OC-SORT tracking options.
- CPU or CUDA inference, with automatic device selection.
- Configurable confidence threshold and input resolution.
- Progress display, video preview, and a Stop button.
- Exports an annotated MP4 and a localization CSV beside the input video.

## Requirements

- Python 3.10 or newer.
- PyQt6, OpenCV, Ultralytics, NumPy, and PyTorch.
- An NVIDIA GPU is optional. CPU inference works but is slower.

## Install

Create and activate a virtual environment.

On Windows PowerShell:

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1

On macOS or Linux:

    python3 -m venv .venv
    source .venv/bin/activate

With the environment activated, install the packages:

    python -m pip install --upgrade pip
    python -m pip install PyQt6 opencv-python ultralytics numpy

Ultralytics installs PyTorch as a dependency. For CUDA, install the PyTorch build that matches your system using the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install the remaining packages above.

## Run

Place fasttrack_pyqt.py in the project folder, activate the virtual environment, then run:

    python fasttrack_pyqt.py

On the first run, Ultralytics may download the default segmentation weights, yolo26n-seg.pt. An internet connection is needed for that initial download.

## Use the app

1. Select an input video with Browse. Supported extensions are MP4, AVI, MOV, MKV, MPEG, and MPG.
2. Keep the model set to yolo26n-seg.pt, or choose another Ultralytics segmentation checkpoint with Segmentation weights.
3. Leave the target class as airplane, or enter a class name present in the chosen model.
4. Choose a tracker and inference settings.
5. Click Segment and track plane. Use Stop to end processing after the current frame.

The default target is airplane. Plane, aircraft, and aeroplane are accepted as aliases. The model must be a segmentation model; detection-only weights cannot generate masks.

## Settings

| Setting | Guidance |
| --- | --- |
| Tracker | ByteTrack is the default fast baseline. BoT-SORT supports moving-camera tracking; OC-SORT is another option for abrupt motion. |
| Input size | 640 is faster. Try 960 or 1280 when the plane is small or distant; larger sizes take longer. |
| Confidence | The default threshold is 0.15. Raise it to reduce weak detections, or lower it if detections are being missed. |
| Device | Auto selects CUDA when available, otherwise CPU. Choose GPU (CUDA) only when PyTorch can access a compatible CUDA GPU. |
| Preview | Controls how often the UI refreshes. Tracking still processes every video frame. |
| m / pixel | Leave at 0 for speed in pixels per second. A positive scale also reports an estimated km/h value. |

## Output files

For an input named flight.mp4, the app writes these files in the same folder:

- flight_airplane_segmented.mp4 — video with the mask, outline, bounding box, center, track ID, trail, confidence, and speed overlays.
- flight_airplane_localization.csv — one row per detected airplane instance per processed frame.

The CSV includes frame and time, track ID, class, confidence, bounding-box pixel coordinates, pixel and normalized center coordinates, mask area, and speed fields. Normalized coordinates are fractions of the frame width or height.

## Sample videos

- [F-16 maneuver clip on Pexels](https://www.pexels.com/video/f-16-18855223/)
- [Close-up airplane clip on Pexels](https://www.pexels.com/video/airplane-flying-through-clear-blue-sky-36254578/)

Download a clip and select the MP4 in the app.

## Tips and limitations

- The default weights are general-purpose. A small or distant aircraft may not receive a reliable mask. Try a higher input size or choose a custom-trained airplane segmentation checkpoint.
- Image-plane speed in pixels per second is a tracking estimate. The km/h value uses the entered m / pixel scale; it is only meaningful when that scale is calibrated for the scene. Camera motion, zoom, and perspective can make real-world speed estimates inaccurate.
- Fast motion, blur, occlusion, or a plane occupying only a few pixels can cause missed detections or a changed track ID.
- For faster processing, use a CUDA-enabled PyTorch installation and start with 640 input size. Changing the preview refresh rate only changes display updates, not inference work.
