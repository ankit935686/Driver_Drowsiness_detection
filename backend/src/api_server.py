from __future__ import annotations

import base64
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from drowsiness_detector import DetectorConfig, DrowsinessDetector


def _build_config(payload: Optional[dict[str, Any]]) -> DetectorConfig:
    payload = payload or {}
    warning_frames = int(payload.get("warning_frames", 5))
    consecutive_frames = int(payload.get("consecutive_frames", 6))
    recovery_frames = int(payload.get("recovery_frames", 3))
    return DetectorConfig(
        ear_threshold=float(payload.get("ear_threshold", 0.224)),
        consecutive_frames=consecutive_frames,
        warning_frames=warning_frames,
        recovery_frames=recovery_frames,
        drowsy_seconds=float(payload.get("drowsy_seconds", consecutive_frames / 10.0)),
        warning_seconds=float(payload.get("warning_seconds", warning_frames / 10.0)),
        recovery_seconds=float(payload.get("recovery_seconds", recovery_frames / 10.0)),
        face_loss_grace_seconds=float(payload.get("face_loss_grace_seconds", 0.25)),
        ear_smoothing_alpha=float(payload.get("ear_smoothing_alpha", 0.35)),
        camera_index=int(payload.get("camera_index", 0)),
        frame_width=int(payload.get("frame_width", 960)),
        frame_height=int(payload.get("frame_height", 540)),
        stage=int(payload.get("stage", 6)),
    )


class DetectionService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._detector: Optional[DrowsinessDetector] = None
        self._running = False
        self._mode = "idle"
        self._last_error: Optional[str] = None
        self._config: Optional[DetectorConfig] = None
        self._last_frame_time: Optional[float] = None

    def start(self, config: DetectorConfig, mode: str = "backend_camera") -> tuple[bool, str]:
        mode = mode if mode in {"backend_camera", "browser"} else "backend_camera"
        with self._lock:
            if self._running:
                return False, "Detection is already running"

            self._running = True
            self._mode = mode
            self._last_error = None
            self._config = config
            self._detector = DrowsinessDetector(config)
            self._last_frame_time = None

            if mode == "backend_camera":
                self._thread = threading.Thread(target=self._run_worker, daemon=True)
                self._thread.start()
                return True, "Detection started (backend camera mode)"

            self._thread = None
            return True, "Detection started (browser frame mode)"

    def stop(self) -> tuple[bool, str]:
        thread_to_join: Optional[threading.Thread] = None
        with self._lock:
            if not self._running:
                return False, "Detection is not running"

            if self._detector is not None:
                self._detector.request_stop()

            thread_to_join = self._thread

        if thread_to_join is not None:
            thread_to_join.join(timeout=5.0)

        with self._lock:
            detector = self._detector
            self._running = False
            self._mode = "idle"
            self._thread = None
            self._detector = None
            self._last_frame_time = None

        if detector is not None:
            with self._process_lock:
                detector.shutdown()

        return True, "Detection stopped"

    def process_browser_frame(
        self, image_data_url: str, include_processed_frame: bool = False
    ) -> tuple[bool, str, Optional[dict[str, Any]], Optional[str]]:
        with self._lock:
            if not self._running:
                return False, "Detection is not running", None, None
            if self._mode != "browser":
                return False, "Detection is not in browser mode", None, None
            detector = self._detector

        if detector is None:
            return False, "Detector is not initialized", None, None

        try:
            if "," in image_data_url:
                image_data_url = image_data_url.split(",", 1)[1]

            frame_bytes = base64.b64decode(image_data_url)
            np_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
            if frame is None:
                return False, "Invalid frame payload", None, None

            now = time.perf_counter()
            with self._lock:
                prev = self._last_frame_time
                self._last_frame_time = now

            fps = None
            if prev is not None:
                fps = 1.0 / max(now - prev, 1e-6)

            with self._process_lock:
                with self._lock:
                    still_running = self._running and self._mode == "browser" and self._detector is detector
                if not still_running:
                    return False, "Detection is not running", None, None

                output = detector.process_external_frame(frame, fps=fps)

            processed_image = None
            if include_processed_frame:
                ok, encoded = cv2.imencode(
                    ".jpg",
                    output,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 70],
                )
                if ok:
                    processed_image = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

            with self._process_lock:
                runtime = detector.get_runtime_status()
            return True, "Frame processed", runtime, processed_image
        except Exception as exc:  # pragma: no cover
            with self._lock:
                self._last_error = str(exc)
            return False, str(exc), None, None

    def status(self) -> dict[str, Any]:
        with self._lock:
            runtime = self._detector.get_runtime_status() if self._detector else None
            return {
                "running": self._running,
                "mode": self._mode,
                "last_error": self._last_error,
                "config": asdict(self._config) if self._config else None,
                "runtime": runtime,
            }

    def _run_worker(self) -> None:
        detector: Optional[DrowsinessDetector]
        with self._lock:
            detector = self._detector

        try:
            if detector is not None:
                detector.run(show_window=False)
        except Exception as exc:  # pragma: no cover
            with self._lock:
                self._last_error = str(exc)
        finally:
            with self._lock:
                self._running = False
                self._mode = "idle"
                self._thread = None
                self._detector = None
                self._last_frame_time = None


FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend" / "app"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/static")
service = DetectionService()


@app.get("/")
def index() -> Any:
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/manifest.json")
def manifest() -> Any:
    return send_from_directory(FRONTEND_DIR, "manifest.json")


@app.get("/service-worker.js")
def service_worker() -> Any:
    return send_from_directory(FRONTEND_DIR, "service-worker.js")


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok", "service": "driver-drowsiness-api"})


@app.post("/detect/start")
def start_detection() -> Any:
    payload = request.get_json(silent=True) or {}
    config = _build_config(payload)
    mode = str(payload.get("mode", "backend_camera"))
    ok, message = service.start(config, mode=mode)
    status_code = 200 if ok else 409
    return jsonify({"ok": ok, "message": message, "status": service.status()}), status_code


@app.post("/detect/stop")
def stop_detection() -> Any:
    ok, message = service.stop()
    status_code = 200 if ok else 409
    return jsonify({"ok": ok, "message": message, "status": service.status()}), status_code


@app.get("/detect/status")
def detection_status() -> Any:
    return jsonify({"ok": True, "status": service.status()})


@app.post("/detect/frame")
def process_detection_frame() -> Any:
    payload = request.get_json(silent=True) or {}
    image = payload.get("image")
    include_processed_frame = bool(payload.get("include_processed_frame", False))
    include_status = bool(payload.get("include_status", False))
    auto_start = bool(payload.get("auto_start", True))
    if not image:
        return jsonify({"ok": False, "message": "Missing image field"}), 400

    if auto_start:
        current_status = service.status()
        if not current_status.get("running"):
            auto_config = _build_config(payload)
            service.start(auto_config, mode="browser")

    ok, message, runtime, processed_image = service.process_browser_frame(
        str(image), include_processed_frame=include_processed_frame
    )
    status_code = 200 if ok else 409
    response = {
        "ok": ok,
        "message": message,
        "runtime": runtime,
        "processed_image": processed_image,
    }
    if include_status:
        response["status"] = service.status()

    return jsonify(
        response
    ), status_code


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
