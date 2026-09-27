"""FastTrack Plane: PyQt6 airplane segmentation, localization, and tracking.

Install:
    python -m pip install PyQt6 opencv-python ultralytics numpy

Run:
    python fasttrack_pyqt.py

Default weights: yolo26n-seg.pt (downloaded by Ultralytics on first run).
For reliable masks on small or distant aircraft, use a custom-trained
Ultralytics segmentation checkpoint (.pt) from the Weights button.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

TRACKERS = {
    "ByteTrack — fastest baseline": "bytetrack.yaml",
    "BoT-SORT — moving camera": "botsort.yaml",
    "OC-SORT — abrupt motion": "ocsort.yaml",
}


def model_class_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def find_class_ids(names: dict[int, str], requested: str) -> list[int]:
    """Resolve airplane/plane/aeroplane aliases against model class names."""
    requested = requested.strip().casefold()
    if not requested:
        return []
    aliases = {
        "plane": "airplane",
        "aircraft": "airplane",
        "aeroplane": "airplane",
    }
    wanted = aliases.get(requested, requested)
    return [
        class_id
        for class_id, label in names.items()
        if aliases.get(label.casefold(), label.casefold()) == wanted
    ]


class TrackingWorker(QThread):
    frame_ready = pyqtSignal(object)
    progress_changed = pyqtSignal(int, int)
    status_changed = pyqtSignal(str)
    completed = pyqtSignal(str, str, int, bool)
    failed = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        model_path: str,
        class_filter: str,
        tracker_config: str,
        image_size: int,
        confidence: float,
        device_choice: str,
        meters_per_pixel: float,
        preview_every: int,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.class_filter = class_filter.strip()
        self.tracker_config = tracker_config
        self.image_size = image_size
        self.confidence = confidence
        self.device_choice = device_choice
        self.meters_per_pixel = meters_per_pixel
        self.preview_every = max(1, preview_every)
        self.stop_event = threading.Event()

    def request_stop(self) -> None:
        self.stop_event.set()

    def _device_args(self) -> tuple[str, int]:
        import torch

        has_cuda = torch.cuda.is_available()
        if self.device_choice == "GPU (CUDA)":
            if not has_cuda:
                raise RuntimeError(
                    "CUDA was selected, but PyTorch cannot access a CUDA GPU. "
                    "Install the GPU build of PyTorch or choose Auto/CPU."
                )
            return "0", 16
        if self.device_choice == "CPU":
            return "cpu", 32
        return ("0", 16) if has_cuda else ("cpu", 32)

    def run(self) -> None:
        writer = None
        stream = None
        rows: list[dict[str, Any]] = []
        failure: str | None = None
        frame_index = 0

        base, _ = os.path.splitext(self.video_path)
        output_video = f"{base}_airplane_segmented.mp4"
        output_csv = f"{base}_airplane_localization.csv"

        try:
            metadata = cv2.VideoCapture(self.video_path)
            try:
                if not metadata.isOpened():
                    raise RuntimeError("OpenCV could not open the selected video.")
                fps = float(metadata.get(cv2.CAP_PROP_FPS))
                total_frames = int(metadata.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                metadata.release()
            if not math.isfinite(fps) or fps <= 0:
                fps = 30.0

            self.status_changed.emit("Loading segmentation model…")
            from ultralytics import YOLO

            model = YOLO(self.model_path)
            if getattr(model, "task", None) != "segment":
                raise ValueError(
                    "Choose segmentation weights, for example yolo26n-seg.pt. "
                    "Detection-only weights cannot produce a plane mask."
                )

            names = model_class_names(model.names)
            classes = find_class_ids(names, self.class_filter)
            if self.class_filter and not classes:
                available = ", ".join(names.values())
                raise ValueError(
                    f"Class '{self.class_filter}' is not in these weights. "
                    f"Available classes: {available}"
                )
            device, quantize = self._device_args()

            self.status_changed.emit(
                f"Tracking {self.class_filter or 'all classes'} at {self.image_size}px on {device}…"
            )
            stream = model.track(
                source=self.video_path,
                stream=True,
                persist=True,
                tracker=self.tracker_config,
                classes=classes or None,
                conf=self.confidence,
                imgsz=self.image_size,
                device=device,
                quantize=quantize,
                max_det=10,
                verbose=False,
                save=False,
            )

            previous: dict[int, tuple[float, float, float]] = {}
            trails: dict[int, deque[tuple[int, int]]] = defaultdict(
                lambda: deque(maxlen=36)
            )
            speed_history: dict[int, deque[float]] = defaultdict(
                lambda: deque(maxlen=5)
            )
            started = time.perf_counter()

            for result in stream:
                if self.stop_event.is_set():
                    break
                frame = result.orig_img
                if frame is None:
                    continue
                height, width = frame.shape[:2]
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("Could not create the output MP4 video.")

                annotated = frame.copy()
                boxes = result.boxes
                masks = result.masks
                if boxes is not None and len(boxes) > 0:
                    box_coords = boxes.xyxy.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    confidences = boxes.conf.cpu().numpy()
                    track_ids = (
                        boxes.id.cpu().numpy().astype(int)
                        if boxes.id is not None
                        else None
                    )

                    for index, coords in enumerate(box_coords):
                        x1, y1, x2, y2 = (float(value) for value in coords)
                        x1i = max(0, min(width - 1, int(round(x1))))
                        y1i = max(0, min(height - 1, int(round(y1))))
                        x2i = max(0, min(width - 1, int(round(x2))))
                        y2i = max(0, min(height - 1, int(round(y2))))
                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0
                        class_id = int(class_ids[index])
                        class_name = names.get(class_id, str(class_id))
                        confidence = float(confidences[index])
                        track_id = int(track_ids[index]) if track_ids is not None else None
                        timestamp = frame_index / fps

                        mask_area = 0
                        if masks is not None and index < len(masks.data):
                            mask = masks.data[index].detach().cpu().numpy().squeeze()
                            if mask.ndim == 2:
                                if mask.shape != (height, width):
                                    mask = cv2.resize(
                                        mask.astype(np.uint8),
                                        (width, height),
                                        interpolation=cv2.INTER_NEAREST,
                                    )
                                binary_mask = mask > 0
                                mask_area = int(np.count_nonzero(binary_mask))
                                if mask_area:
                                    tint = np.array([70, 220, 90], dtype=np.float32)
                                    annotated[binary_mask] = (
                                        annotated[binary_mask].astype(np.float32) * 0.62
                                        + tint * 0.38
                                    ).astype(np.uint8)
                                    contours, _ = cv2.findContours(
                                        binary_mask.astype(np.uint8),
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE,
                                    )
                                    cv2.drawContours(
                                        annotated, contours, -1, (0, 255, 255), 2
                                    )

                        speed_px_s = None
                        speed_kmh = None
                        display_speed = ""
                        if track_id is not None:
                            old = previous.get(track_id)
                            if old is not None:
                                old_x, old_y, old_time = old
                                elapsed = timestamp - old_time
                                if elapsed > 0:
                                    speed_px_s = math.hypot(
                                        center_x - old_x, center_y - old_y
                                    ) / elapsed
                                    speed_history[track_id].append(speed_px_s)
                                    smoothed = sum(speed_history[track_id]) / len(
                                        speed_history[track_id]
                                    )
                                    display_speed = f"{smoothed:.0f} px/s"
                                    if self.meters_per_pixel > 0:
                                        speed_kmh = (
                                            smoothed * self.meters_per_pixel * 3.6
                                        )
                                        display_speed = f"{speed_kmh:.1f} km/h"
                            previous[track_id] = (center_x, center_y, timestamp)
                            trails[track_id].append(
                                (int(round(center_x)), int(round(center_y)))
                            )
                            points = list(trails[track_id])
                            if len(points) > 1:
                                cv2.polylines(
                                    annotated,
                                    [np.asarray(points, dtype=np.int32)],
                                    False,
                                    (255, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )

                        cv2.rectangle(
                            annotated, (x1i, y1i), (x2i, y2i), (255, 170, 0), 2
                        )
                        center_pt = (int(round(center_x)), int(round(center_y)))
                        cv2.drawMarker(
                            annotated,
                            center_pt,
                            (255, 255, 255),
                            cv2.MARKER_CROSS,
                            14,
                            2,
                        )
                        id_text = "" if track_id is None else f" ID:{track_id}"
                        label = f"{class_name}{id_text} {confidence:.2f}"
                        if display_speed:
                            label += f" | {display_speed}"
                        label_y = max(22, y1i - 8)
                        cv2.putText(
                            annotated,
                            label,
                            (x1i, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.62,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        rows.append(
                            {
                                "frame": frame_index,
                                "time_s": round(timestamp, 6),
                                "track_id": "" if track_id is None else track_id,
                                "class_name": class_name,
                                "confidence": round(confidence, 5),
                                "bbox_x1_px": round(x1, 2),
                                "bbox_y1_px": round(y1, 2),
                                "bbox_x2_px": round(x2, 2),
                                "bbox_y2_px": round(y2, 2),
                                "center_x_px": round(center_x, 2),
                                "center_y_px": round(center_y, 2),
                                "center_x_norm": round(center_x / width, 6),
                                "center_y_norm": round(center_y / height, 6),
                                "mask_area_px": mask_area,
                                "speed_px_s": (
                                    "" if speed_px_s is None else round(speed_px_s, 2)
                                ),
                                "speed_kmh": (
                                    "" if speed_kmh is None else round(speed_kmh, 2)
                                ),
                            }
                        )

                writer.write(annotated)
                frame_index += 1
                if frame_index == 1 or frame_index % self.preview_every == 0:
                    self.frame_ready.emit(annotated)
                self.progress_changed.emit(frame_index, total_frames)
                if frame_index % 10 == 0:
                    elapsed = max(0.001, time.perf_counter() - started)
                    self.status_changed.emit(
                        f"Processed {frame_index}/{total_frames or '?'} frames "
                        f"at {frame_index / elapsed:.1f} inference FPS"
                    )

            if frame_index == 0:
                raise RuntimeError("No video frames were processed.")

        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            if writer is not None:
                writer.release()
            if stream is not None and hasattr(stream, "close"):
                stream.close()

        if failure:
            self.failed.emit(failure)
            return

        try:
            fields = [
                "frame",
                "time_s",
                "track_id",
                "class_name",
                "confidence",
                "bbox_x1_px",
                "bbox_y1_px",
                "bbox_x2_px",
                "bbox_y2_px",
                "center_x_px",
                "center_y_px",
                "center_x_norm",
                "center_y_norm",
                "mask_area_px",
                "speed_px_s",
                "speed_kmh",
            ]
            with open(output_csv, "w", newline="", encoding="utf-8") as file:
                csv_writer = csv.DictWriter(file, fieldnames=fields)
                csv_writer.writeheader()
                csv_writer.writerows(rows)
            self.completed.emit(
                output_video,
                output_csv,
                len(rows),
                self.stop_event.is_set(),
            )
        except Exception as exc:
            self.failed.emit(f"Could not save localization CSV: {exc}")


class FastTrackWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FastTrack — Airplane Segmentation and Tracking")
        self.resize(1180, 820)
        self.worker: TrackingWorker | None = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(11)

        title = QLabel("Airplane Segmentation & Tracking")
        title.setStyleSheet("font-size: 25px; font-weight: 700; color: #e8edf5;")
        subtitle = QLabel(
            "Find the plane in each frame, draw its mask and box, and export its location."
        )
        subtitle.setStyleSheet("color: #aab4c4;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(9)

        video_row = QHBoxLayout()
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("Select a video file")
        video_button = QPushButton("Browse…")
        video_button.clicked.connect(self.choose_video)
        video_row.addWidget(self.video_edit, 1)
        video_row.addWidget(video_button)
        form.addRow("Input video", video_row)

        model_row = QHBoxLayout()
        self.model_edit = QLineEdit("yolo26n-seg.pt")
        self.model_edit.setToolTip(
            "Use a segmentation checkpoint, not detection-only weights."
        )
        model_button = QPushButton("Segmentation weights…")
        model_button.clicked.connect(self.choose_model)
        model_row.addWidget(self.model_edit, 1)
        model_row.addWidget(model_button)
        form.addRow("Model", model_row)

        tracking_options = QHBoxLayout()
        self.class_edit = QLineEdit("airplane")
        self.class_edit.setPlaceholderText("airplane / plane; blank tracks all classes")

        self.tracker_combo = QComboBox()
        for label, config in TRACKERS.items():
            self.tracker_combo.addItem(label, config)

        self.image_size_combo = QComboBox()
        self.image_size_combo.addItem("640 — faster", 640)
        self.image_size_combo.addItem("960 — small / distant plane", 960)
        self.image_size_combo.addItem("1280 — slower / fine detail", 1280)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.01, 0.99)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.15)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (GPU if available)", "auto")
        self.device_combo.addItem("CPU", "CPU")
        self.device_combo.addItem("GPU (CUDA)", "GPU (CUDA)")

        self.meters_per_pixel = QDoubleSpinBox()
        self.meters_per_pixel.setRange(0.0, 1000000.0)
        self.meters_per_pixel.setDecimals(8)
        self.meters_per_pixel.setSingleStep(0.0001)
        self.meters_per_pixel.setValue(0.0)
        self.meters_per_pixel.setToolTip(
            "Leave at 0 to report image-plane speed in pixels/second."
        )

        self.preview_every_combo = QComboBox()
        self.preview_every_combo.addItem("Every frame", 1)
        self.preview_every_combo.addItem("Every 2 frames — smoother processing", 2)
        self.preview_every_combo.addItem("Every 3 frames", 3)
        self.preview_every_combo.setCurrentIndex(1)

        for label, widget in (
            ("Target", self.class_edit),
            ("Tracker", self.tracker_combo),
            ("Input size", self.image_size_combo),
            ("Confidence", self.confidence_spin),
        ):
            tracking_options.addWidget(QLabel(label))
            tracking_options.addWidget(widget)
        tracking_options.addStretch(1)
        performance_options = QHBoxLayout()
        for label, widget in (
            ("Device", self.device_combo),
            ("m / pixel", self.meters_per_pixel),
            ("Preview", self.preview_every_combo),
        ):
            performance_options.addWidget(QLabel(label))
            performance_options.addWidget(widget)
        performance_options.addStretch(1)
        form.addRow("Detection", tracking_options)
        form.addRow("Performance / speed", performance_options)
        layout.addLayout(form)

        self.preview = QLabel("Select a video and start airplane tracking")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(720, 400)
        self.preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.preview.setStyleSheet(
            "background: #10151d; color: #95a1b2; border: 1px solid #354052; "
            "border-radius: 8px;"
        )
        layout.addWidget(self.preview, 1)

        controls = QHBoxLayout()
        self.start_button = QPushButton("Segment and track plane")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.start_tracking)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_tracking)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(250)
        controls.addWidget(self.progress)
        layout.addLayout(controls)

        self.status_label = QLabel("Green fill/yellow outline: plane mask · orange box: localization")
        self.status_label.setStyleSheet("color: #bac4d2;")
        layout.addWidget(self.status_label)
        self.output_label = QLabel("")
        self.output_label.setWordWrap(True)
        self.output_label.setStyleSheet("color: #91d9ad;")
        layout.addWidget(self.output_label)

        note = QLabel(
            "For a tiny or distant plane, try 960 input size or custom-trained plane "
            "segmentation weights. Larger input sizes take longer."
        )
        note.setStyleSheet("color: #aab4c4;")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.setStyleSheet(
            "QMainWindow, QWidget { background: #171d27; color: #e6ebf2; }"
            "QLineEdit, QComboBox, QDoubleSpinBox { background: #222b38; "
            "border: 1px solid #39475a; border-radius: 5px; padding: 6px; }"
            "QPushButton { background: #2b3747; border: 1px solid #45556b; "
            "border-radius: 5px; padding: 8px 12px; }"
            "QPushButton:hover { background: #36465b; }"
            "QPushButton:disabled { color: #738094; background: #222b38; }"
            "QPushButton#startButton { background: #2364d2; border-color: #3479ed; "
            "font-weight: 600; }"
            "QPushButton#startButton:hover { background: #2f75eb; }"
            "QProgressBar { background: #222b38; border: 1px solid #39475a; "
            "border-radius: 4px; text-align: center; }"
            "QProgressBar::chunk { background: #3982ec; border-radius: 3px; }"
        )

    def choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose airplane video",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.mpeg *.mpg);;All files (*.*)",
        )
        if path:
            self.video_edit.setText(path)

    def choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose segmentation weights",
            "",
            "PyTorch weights (*.pt);;All files (*.*)",
        )
        if path:
            self.model_edit.setText(path)

    def start_tracking(self) -> None:
        video_path = self.video_edit.text().strip()
        model_path = self.model_edit.text().strip()
        if not video_path or not os.path.isfile(video_path):
            QMessageBox.warning(self, "Video required", "Choose a video file first.")
            return
        if not model_path:
            QMessageBox.warning(self, "Model required", "Enter or choose segmentation weights.")
            return

        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.output_label.clear()
        self.preview.setText("Loading model and video…")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.worker = TrackingWorker(
            video_path=video_path,
            model_path=model_path,
            class_filter=self.class_edit.text(),
            tracker_config=str(self.tracker_combo.currentData()),
            image_size=int(self.image_size_combo.currentData()),
            confidence=self.confidence_spin.value(),
            device_choice=str(self.device_combo.currentData()),
            meters_per_pixel=self.meters_per_pixel.value(),
            preview_every=int(self.preview_every_combo.currentData()),
        )
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.progress_changed.connect(self.update_progress)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.completed.connect(self.tracking_completed)
        self.worker.failed.connect(self.tracking_failed)
        self.worker.finished.connect(self.worker_stopped)
        self.worker.start()

    def stop_tracking(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.stop_button.setEnabled(False)
            self.status_label.setText("Stopping after the current frame…")

    def show_frame(self, frame: Any) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)

    def update_progress(self, current: int, total: int) -> None:
        if total > 0:
            self.progress.setValue(min(100, int(100 * current / total)))
        else:
            self.progress.setRange(0, 0)

    def tracking_completed(
        self, output_video: str, output_csv: str, detection_count: int, was_stopped: bool
    ) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        suffix = "Stopped early" if was_stopped else "Finished"
        self.status_label.setText(
            f"{suffix}. Localized {detection_count} plane instance(s) across processed frames."
        )
        self.output_label.setText(
            f"Segmented video: {output_video}\nLocalization CSV: {output_csv}"
        )

    def tracking_failed(self, message: str) -> None:
        self.progress.setRange(0, 100)
        self.status_label.setText("Tracking failed")
        QMessageBox.critical(self, "Tracking error", message)

    def worker_stopped(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.worker = None

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API name
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FastTrackWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
