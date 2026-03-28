from __future__ import annotations

import argparse

from drowsiness_detector import DetectorConfig, DrowsinessDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detector")
    parser.add_argument("--stage", type=int, default=6, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--ear-threshold", type=float, default=0.224)
    parser.add_argument("--consecutive-frames", type=int, default=23)
    parser.add_argument("--warning-frames", type=int, default=14)
    parser.add_argument("--recovery-frames", type=int, default=3)
    parser.add_argument("--ear-smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--open-seconds", type=float, default=8.0)
    parser.add_argument("--closed-seconds", type=float, default=4.0)
    parser.add_argument("--decision-seconds", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DetectorConfig(
        ear_threshold=args.ear_threshold,
        consecutive_frames=args.consecutive_frames,
        warning_frames=args.warning_frames,
        recovery_frames=args.recovery_frames,
        ear_smoothing_alpha=args.ear_smoothing_alpha,
        camera_index=args.camera_index,
        frame_width=args.width,
        frame_height=args.height,
        stage=args.stage,
    )
    detector = DrowsinessDetector(config)
    if args.calibrate:
        threshold, frames = detector.calibrate(
            open_seconds=args.open_seconds,
            closed_seconds=args.closed_seconds,
            decision_seconds=args.decision_seconds,
        )
        print("Apply suggested values with:")
        print(
            "python backend/src/main.py --stage 6 "
            f"--ear-threshold {threshold:.3f} "
            f"--consecutive-frames {frames}"
        )
    elif args.self_test:
        detector.self_test(frames=args.frames)
    else:
        detector.run()


if __name__ == "__main__":
    main()
