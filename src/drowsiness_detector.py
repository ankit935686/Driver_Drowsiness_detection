from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist


@dataclass
class DetectorConfig:
    ear_threshold: float = 0.25
    consecutive_frames: int = 20
    warning_frames: int = 12
    recovery_frames: int = 3
    ear_smoothing_alpha: float = 0.35
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    stage: int = 6


class AlertPlayer:
    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _play_loop(self) -> None:
        if platform.system().lower().startswith("win"):
            import winsound

            while not self._stop_event.is_set():
                winsound.Beep(2000, 180)
                time.sleep(0.05)
            return

        while not self._stop_event.is_set():
            print("\a", end="", flush=True)
            time.sleep(0.2)


class DrowsinessDetector:
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.frame_counter = 0
        self.open_frame_counter = 0
        self.is_drowsy = False
        self.is_warning = False
        self.smoothed_ear: Optional[float] = None
        self.alert = AlertPlayer()

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = None
        if self.config.stage >= 2:
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    @staticmethod
    def eye_aspect_ratio(eye: np.ndarray) -> float:
        vertical_1 = dist.euclidean(eye[1], eye[5])
        vertical_2 = dist.euclidean(eye[2], eye[4])
        horizontal = dist.euclidean(eye[0], eye[3])

        if horizontal == 0:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def _extract_eye_points(
        self, face_landmarks: object, frame_shape: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame_shape[:2]

        def to_xy(index: int) -> Tuple[int, int]:
            landmark = face_landmarks.landmark[index]
            return int(landmark.x * w), int(landmark.y * h)

        left_eye = np.array([to_xy(i) for i in self.LEFT_EYE_IDX], dtype=np.float32)
        right_eye = np.array([to_xy(i) for i in self.RIGHT_EYE_IDX], dtype=np.float32)
        return left_eye, right_eye

    def _draw_eye_contour(self, frame: np.ndarray, eye: np.ndarray) -> None:
        points = eye.astype(np.int32)
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 255), thickness=1)

    def _compute_avg_ear(self, frame: np.ndarray) -> Optional[float]:
        if self._face_mesh is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        face_landmarks = result.multi_face_landmarks[0]
        left_eye, right_eye = self._extract_eye_points(face_landmarks, frame.shape)
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        return (left_ear + right_ear) / 2.0

    def _update_state(self, avg_ear: float) -> None:
        if avg_ear < self.config.ear_threshold:
            self.open_frame_counter = 0
            self.frame_counter += 1
            if self.frame_counter >= self.config.consecutive_frames:
                self.is_drowsy = True
                self.is_warning = True
                if self.config.stage >= 6:
                    self.alert.start()
            elif self.frame_counter >= self.config.warning_frames:
                self.is_warning = True
                self.is_drowsy = False
                self.alert.stop()
        else:
            self.open_frame_counter += 1
            if self.open_frame_counter >= self.config.recovery_frames:
                self.frame_counter = 0
                self.is_drowsy = False
                self.is_warning = False
                self.alert.stop()

    def _overlay_runtime_stats(self, frame: np.ndarray, fps: float) -> None:
        cv2.putText(
            frame,
            f"Step {self.config.stage}/6",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Threshold: {self.config.ear_threshold:.3f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.config.stage == 1:
            cv2.putText(
                frame,
                "Step 1: Camera capture only",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            return frame

        if self._face_mesh is None:
            return frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]
            if self.config.stage == 2:
                h, w = frame.shape[:2]
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
                cv2.putText(
                    frame,
                    "Step 2: Face Mesh landmarks",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                return frame

            left_eye, right_eye = self._extract_eye_points(face_landmarks, frame.shape)
            self._draw_eye_contour(frame, left_eye)
            self._draw_eye_contour(frame, right_eye)

            if self.config.stage == 3:
                cv2.putText(
                    frame,
                    "Step 3: Eye landmarks extracted",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                return frame

            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if self.smoothed_ear is None:
                self.smoothed_ear = avg_ear
            else:
                alpha = min(1.0, max(0.0, self.config.ear_smoothing_alpha))
                self.smoothed_ear = (alpha * avg_ear) + ((1.0 - alpha) * self.smoothed_ear)

            cv2.putText(
                frame,
                f"EAR: {avg_ear:.3f}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"EAR(s): {self.smoothed_ear:.3f}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 255, 180),
                2,
                cv2.LINE_AA,
            )

            if self.config.stage == 4:
                cv2.putText(
                    frame,
                    "Step 4: EAR computation",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                return frame

            self._update_state(self.smoothed_ear)

            cv2.putText(
                frame,
                f"Closed frames: {self.frame_counter}/{self.config.consecutive_frames}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if self.is_drowsy:
                state_text = "State: DROWSY"
                state_color = (0, 0, 255)
            elif self.is_warning:
                state_text = "State: WARNING"
                state_color = (0, 215, 255)
            else:
                state_text = "State: AWAKE"
                state_color = (0, 255, 0)
            cv2.putText(
                frame,
                state_text,
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                state_color,
                2,
                cv2.LINE_AA,
            )

            if self.config.stage == 5:
                cv2.putText(
                    frame,
                    "Step 5: Drowsiness logic",
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                return frame
        else:
            self.frame_counter = 0
            self.is_drowsy = False
            self.alert.stop()
            cv2.putText(
                frame,
                "No face detected",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        if self.config.stage >= 6 and self.is_drowsy:
            cv2.putText(
                frame,
                "DROWSINESS ALERT!",
                (20, 255),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        return frame

    def self_test(self, frames: int = 120) -> None:
        cap = cv2.VideoCapture(self.config.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        if not cap.isOpened():
            raise RuntimeError("Could not access the webcam")

        try:
            start = time.perf_counter()
            count = 0
            while count < frames:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Frame capture failed at frame {count + 1}")

                frame = cv2.flip(frame, 1)
                self.process_frame(frame)
                count += 1

            elapsed = time.perf_counter() - start
            fps = count / max(elapsed, 1e-6)
            print(
                f"Self-test passed (step {self.config.stage}): "
                f"{count} frames in {elapsed:.2f}s (~{fps:.1f} FPS)."
            )
        finally:
            self.alert.stop()
            cap.release()
            if self._face_mesh is not None:
                self._face_mesh.close()

    def calibrate(
        self,
        open_seconds: float = 8.0,
        closed_seconds: float = 4.0,
        decision_seconds: float = 0.8,
    ) -> Tuple[float, int]:
        if self._face_mesh is None:
            raise RuntimeError("Calibration requires stage 2 or higher")

        cap = cv2.VideoCapture(self.config.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        if not cap.isOpened():
            raise RuntimeError("Could not access the webcam")

        open_samples = []
        closed_samples = []
        total_frames = 0
        total_start = time.perf_counter()

        phases = [
            ("open", "Keep eyes open naturally", max(open_seconds, 1.0)),
            ("closed", "Gently close your eyes", max(closed_seconds, 1.0)),
        ]

        try:
            for phase_name, message, duration in phases:
                phase_start = time.perf_counter()

                while True:
                    now = time.perf_counter()
                    elapsed = now - phase_start
                    if elapsed >= duration:
                        break

                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Calibration failed: frame capture error")

                    frame = cv2.flip(frame, 1)
                    avg_ear = self._compute_avg_ear(frame)
                    if avg_ear is not None:
                        if phase_name == "open":
                            open_samples.append(avg_ear)
                        else:
                            closed_samples.append(avg_ear)

                    remaining = max(0.0, duration - elapsed)
                    cv2.putText(
                        frame,
                        f"Calibration: {phase_name.upper()} eyes",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        message,
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Remaining: {remaining:.1f}s",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    if avg_ear is not None:
                        cv2.putText(
                            frame,
                            f"EAR: {avg_ear:.3f}",
                            (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        cv2.putText(
                            frame,
                            "Face not detected",
                            (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 165, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    cv2.putText(
                        frame,
                        "Press q to cancel calibration",
                        (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (200, 200, 200),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.imshow("Driver Drowsiness Calibration", frame)
                    total_frames += 1

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise RuntimeError("Calibration cancelled by user")

            total_elapsed = max(time.perf_counter() - total_start, 1e-6)
            measured_fps = total_frames / total_elapsed

            if len(open_samples) < 10 or len(closed_samples) < 5:
                raise RuntimeError(
                    "Not enough EAR samples. Ensure your face stays visible during calibration."
                )

            open_mean = float(np.mean(open_samples))
            closed_mean = float(np.mean(closed_samples))
            open_med = float(np.median(open_samples))
            closed_med = float(np.median(closed_samples))

            open_ref = open_med
            closed_ref = closed_med

            if open_ref > closed_ref:
                suggested_threshold = (open_ref + closed_ref) / 2.0
            else:
                suggested_threshold = open_ref * 0.85

            separation = open_ref - closed_ref
            quality_ok = separation >= 0.03

            suggested_frames = max(1, int(round(measured_fps * max(0.2, decision_seconds))))

            print("Calibration complete")
            print(f"Open-eye EAR mean:   {open_mean:.3f}")
            print(f"Closed-eye EAR mean: {closed_mean:.3f}")
            print(f"Open-eye EAR median:   {open_med:.3f}")
            print(f"Closed-eye EAR median: {closed_med:.3f}")
            print(f"Measured FPS:        {measured_fps:.1f}")
            print(f"Suggested threshold: {suggested_threshold:.3f}")
            print(f"Suggested frames:    {suggested_frames}")
            if not quality_ok:
                print("WARNING: Open/closed EAR separation is too small for reliable detection.")
                print("Retry calibration with better lighting and keep head steady.")

            return suggested_threshold, suggested_frames
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self._face_mesh is not None:
                self._face_mesh.close()

    def run(self) -> None:
        cap = cv2.VideoCapture(self.config.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        if not cap.isOpened():
            raise RuntimeError("Could not access the webcam")

        try:
            prev = time.perf_counter()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)

                now = time.perf_counter()
                fps = 1.0 / max(now - prev, 1e-6)
                prev = now

                output = self.process_frame(frame)
                self._overlay_runtime_stats(output, fps)
                cv2.imshow("Driver Drowsiness Detection", output)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.alert.stop()
            cap.release()
            cv2.destroyAllWindows()
            if self._face_mesh is not None:
                self._face_mesh.close()
