# Driver Drowsiness Detection System

Real-time driver drowsiness detection using EAR (Eye Aspect Ratio), OpenCV, MediaPipe Face Mesh, and a Flask dashboard.

This project supports two operating modes:

- Native OpenCV mode (fastest, direct webcam processing)
- Browser mode (camera in browser, frames processed by backend API)

## Table of Contents

- Overview
- Features
- Project Structure
- Requirements
- Installation
- Quick Start
- Run Modes
- Detection Timing Model
- Calibration
- API Reference
- Testing Checklist
- Troubleshooting

## Overview

The detector tracks eye landmarks, computes EAR values, smooths them, and triggers:

- WARNING when eye closure duration crosses warning threshold
- DROWSY alert when closure duration crosses drowsy threshold

The current implementation uses time-based thresholds for more stable behavior under variable FPS.

## Features

- MediaPipe Face Mesh based eye landmark extraction
- EAR and smoothed EAR tracking
- Time-based warning and drowsy thresholds
- Non-blocking alert sound
- Stage-wise detector workflow (1 through 6)
- Browser dashboard with camera preview and live backend detection frame
- Flask API for start, stop, status, and frame processing

## Project Structure

```text
backend/
	requirements.txt
	src/
		main.py
		drowsiness_detector.py
		api_server.py
frontend/
	app/
		index.html
		app.js
		style.css
		manifest.json
		service-worker.js
README.md
```

## Requirements

- Python 3.10+
- Webcam
- Windows, Linux, or macOS

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

## Quick Start

Run native OpenCV detector:

```bash
python backend/src/main.py --stage 6
```

Press q in OpenCV window to quit.

Run Flask API + dashboard:

```bash
python backend/src/api_server.py
```

Open:

```text
http://127.0.0.1:5000/
```

## Run Modes

### 1. Native OpenCV Mode (recommended for max responsiveness)

```bash
python backend/src/main.py --stage 6 --warning-seconds 0.5 --drowsy-seconds 0.6 --recovery-seconds 0.3
```

### 2. Browser Mode (presentation/dashboard mode)

1. Start API:

```bash
python backend/src/api_server.py
```

2. Open dashboard at http://127.0.0.1:5000/
3. Select Browser Camera mode
4. Click Start Detection

## Detection Timing Model

The system now supports explicit time-based thresholds:

- warning_seconds
- drowsy_seconds
- recovery_seconds
- face_loss_grace_seconds

UI frame fields are mapped to seconds in browser mode for convenience:

- warning_seconds = warning_frames / 10
- drowsy_seconds = consecutive_frames / 10
- recovery_seconds = recovery_frames / 10

Recommended safety defaults:

- warning_seconds: 0.5
- drowsy_seconds: 0.6
- recovery_seconds: 0.3
- ear_threshold: 0.224 (adjust per person)

## Calibration

Run guided calibration:

```bash
python backend/src/main.py --calibrate --stage 6
```

Optional calibration timings:

```bash
python backend/src/main.py --calibrate --stage 6 --open-seconds 8 --closed-seconds 4 --decision-seconds 0.8
```

After calibration, apply suggested threshold and decision window values.

## API Reference

Base URL:

```text
http://127.0.0.1:5000
```

Health:

```bash
curl http://127.0.0.1:5000/health
```

Start detection:

```bash
curl -X POST http://127.0.0.1:5000/detect/start \
	-H "Content-Type: application/json" \
	-d "{\"mode\":\"browser\",\"warning_seconds\":0.5,\"drowsy_seconds\":0.6}"
```

Stop detection:

```bash
curl -X POST http://127.0.0.1:5000/detect/stop
```

Get status:

```bash
curl http://127.0.0.1:5000/detect/status
```

Process browser frame:

```bash
curl -X POST http://127.0.0.1:5000/detect/frame \
	-H "Content-Type: application/json" \
	-d "{\"image\":\"data:image/jpeg;base64,...\",\"include_processed_frame\":false}"
```

## Testing Checklist

### OpenCV Mode

1. Run main.py stage 6
2. Keep eyes open and confirm state stays AWAKE
3. Close eyes for about 0.5s and confirm WARNING
4. Keep eyes closed for about 0.6s and confirm DROWSY alert

### Browser Mode

1. Start api_server.py
2. Open dashboard and click Start Detection
3. Confirm FPS updates and state changes
4. Repeat warning and drowsy timing checks
5. Click Stop Detection and verify clean stop without server errors

## Troubleshooting

### Server not starting from project root

Run from backend/src or use full path:

```bash
python backend/src/api_server.py
```

### Repeated 409 on /detect/frame

Detection is not running yet. Click Start Detection or call /detect/start first.

### Browser feels less sensitive than OpenCV

Browser mode includes encode/upload/decode overhead. To improve responsiveness:

- Keep browser tab active and visible
- Use lower camera resolution in dashboard
- Keep include_processed_frame disabled except for periodic preview updates
- Prefer native OpenCV mode for strict latency testing

### Stale frontend after updates

Hard refresh:

```text
Ctrl+F5
```

### Camera conflict

If another app is using camera, close it and retry.
