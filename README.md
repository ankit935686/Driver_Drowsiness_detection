# Driver Drowsiness Detection (Core)

This phase includes only the standalone Python detection system.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

Press `q` to quit.

## Core Step-by-Step Workflow (Single Runner)

Use one command path for all stages:

```bash
python src/main.py --stage <1-6>
```

Continuous validation at any stage:

```bash
python src/main.py --stage <1-6> --self-test --frames 120
```

Examples:

```bash
# Step 1: camera only
python src/main.py --stage 1

# Step 2: face mesh landmarks
python src/main.py --stage 2

# Step 4 self-test: camera + landmarks + EAR
python src/main.py --stage 4 --self-test --frames 120
```

If you have multiple cameras:

```bash
python src/main.py --stage 1 --camera-index 1
```

## Calibration (Recommended Before Final Stage 6 Run)

Run guided calibration to estimate personalized EAR threshold and frame window:

```bash
python src/main.py --calibrate --stage 6
```

Note: `--calibrate` does not run drowsiness alert detection. It only measures EAR and suggests values.

Optional custom calibration durations:

```bash
python src/main.py --calibrate --stage 6 --open-seconds 8 --closed-seconds 4 --decision-seconds 0.8
```

After calibration, use the printed command with suggested values, for example:

```bash
python src/main.py --stage 6 --ear-threshold 0.24 --consecutive-frames 24
```

If detection feels delayed while testing, temporarily lower the frame window for validation:

```bash
python src/main.py --stage 6 --ear-threshold 0.224 --consecutive-frames 12
```

## Step-by-step build plan

1. Stage 1: Webcam capture and live frame display
2. Stage 2: Face Mesh landmark detection
3. Stage 3: Eye landmark extraction (left and right eyes)
4. Stage 4: EAR calculation
5. Stage 5: Drowsiness logic using threshold + consecutive frame counter
6. Stage 6: On-screen alert and non-blocking sound alert
