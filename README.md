# Driver Drowsiness Detection (Core)

This phase includes only the standalone Python detection system.

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
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

## Run

```bash
python backend/src/main.py
```

Press `q` to quit.

## Core Step-by-Step Workflow (Single Runner)

Use one command path for all stages:

```bash
python backend/src/main.py --stage <1-6>
```

Continuous validation at any stage:

```bash
python backend/src/main.py --stage <1-6> --self-test --frames 120
```

Examples:

```bash
# Step 1: camera only
python backend/src/main.py --stage 1

# Step 2: face mesh landmarks
python backend/src/main.py --stage 2

# Step 4 self-test: camera + landmarks + EAR
python backend/src/main.py --stage 4 --self-test --frames 120
```

If you have multiple cameras:

```bash
python backend/src/main.py --stage 1 --camera-index 1
```

## Calibration (Recommended Before Final Stage 6 Run)

Run guided calibration to estimate personalized EAR threshold and frame window:

```bash
python backend/src/main.py --calibrate --stage 6
```

Note: `--calibrate` does not run drowsiness alert detection. It only measures EAR and suggests values.

Optional custom calibration durations:

```bash
python backend/src/main.py --calibrate --stage 6 --open-seconds 8 --closed-seconds 4 --decision-seconds 0.8
```

After calibration, use the printed command with suggested values, for example:

```bash
python backend/src/main.py --stage 6 --ear-threshold 0.24 --consecutive-frames 24
```

If detection feels delayed while testing, temporarily lower the frame window for validation:

```bash
python backend/src/main.py --stage 6 --ear-threshold 0.224 --consecutive-frames 12
```

## Step-by-step build plan

1. Stage 1: Webcam capture and live frame display
2. Stage 2: Face Mesh landmark detection
3. Stage 3: Eye landmark extraction (left and right eyes)
4. Stage 4: EAR calculation
5. Stage 5: Drowsiness logic using threshold + consecutive frame counter
6. Stage 6: On-screen alert and non-blocking sound alert

## Phase 2 (Flask API)

Run the backend API:

```bash
python backend/src/api_server.py
```

Open the direct test UI in browser:

```text
http://127.0.0.1:5000/
```

From the UI you can:
- Start detection in browser-camera mode (getUserMedia)
- Stop detection
- View local preview and send frames to backend
- View live runtime status (state, EAR, FPS, closed frames)
- Refresh and inspect raw JSON payload

Health check:

```bash
curl http://127.0.0.1:5000/health
```

Start detection (choose mode: `browser` or `backend_camera`):

```bash
curl -X POST http://127.0.0.1:5000/detect/start -H "Content-Type: application/json" -d "{\"mode\":\"browser\"}"
```

Start detection with custom tuning values:

```bash
curl -X POST http://127.0.0.1:5000/detect/start -H "Content-Type: application/json" -d "{\"ear_threshold\":0.224,\"consecutive_frames\":23}"
```

Stop detection:

```bash
curl -X POST http://127.0.0.1:5000/detect/stop
```

Get live detection status (for frontend polling):

```bash
curl http://127.0.0.1:5000/detect/status
```

Send a browser frame for processing (used by UI loop):

```bash
curl -X POST http://127.0.0.1:5000/detect/frame -H "Content-Type: application/json" -d "{\"image\":\"data:image/jpeg;base64,...\"}"
```
