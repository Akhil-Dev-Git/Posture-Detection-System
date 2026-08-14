# Posture Detection System

AI-powered real-time surveillance system with human posture detection, action recognition, and violence detection capabilities.

## Features

- **Pose Estimation** — MediaPipe BlazePose with 33 skeletal keypoints
- **Posture Classification** — Hybrid rule-based + ML (standing, sitting, bending, raising hands, jumping, lying down)
- **Action Recognition** — LSTM-based temporal sequence classification (8 action classes)
- **Violence Detection** — MobileNetV2 + Bi-LSTM with temporal smoothing
- **Real-Time Dashboard** — Flask web app with live streaming, alerts, and statistics
- **Multi-Channel Alerts** — Visual overlay, audio, email (SMTP), SMS (Twilio)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run standalone webcam demo
python demo.py --camera 0

# 3. Run with web dashboard
python app.py
# Open http://localhost:5000 in browser
```

## Commands

| Command | Description |
|---------|-------------|
| `python demo.py` | Webcam demo with OpenCV window |
| `python app.py` | Web dashboard with MJPEG streaming |
| `python training/collect_pose_data.py --action standing` | Collect training data |
| `python training/train_action_model.py` | Train LSTM action model |
| `python training/train_violence_model.py --data /path/to/dataset` | Train violence model |
| `python training/data_augmentation.py --dir datasets/pose_sequences/standing` | Augment data |
| `pytest tests/` | Run all tests |

## Architecture

```
Camera Input -> Pose Estimation (MediaPipe)
                  |
                  +-> Posture Classification (Rules + ML)
                  +-> Action Recognition (LSTM)
                  +-> Violence Detection (MobileNetV2 + Bi-LSTM)
                  +-> Alert Manager (Visual + Audio + Email + SMS)
                  +-> Flask Dashboard (Real-time stats)
```

## Detection Classes

### Postures (6)
Standing, Sitting, Bending, Raising Hands, Jumping, Lying Down

### Actions (8)
Standing, Walking, Sitting Down, Standing Up, Raising Hand, Jumping, Waving, Pointing

### Violence (2)
Violent / Non-violent

## Project Structure

```
posture-detection-system/
├── app.py                    # Flask web dashboard
├── demo.py                   # Standalone webcam demo
├── config.py                 # All configuration constants
├── requirements.txt          # Python dependencies
├── src/                      # Core modules
│   ├── pose_estimator.py
│   ├── posture_classifier.py
│   ├── action_recognizer.py
│   ├── violence_detector.py
│   ├── alert_manager.py
│   ├── video_processor.py
│   └── utils.py
├── training/                  # Data collection & training
│   ├── collect_pose_data.py
│   ├── train_action_model.py
│   ├── train_violence_model.py
│   └── data_augmentation.py
├── datasets/                  # Training data
├── models/                    # Saved models
├── tests/                     # Unit tests
├── templates/                 # Flask HTML
└── static/                    # CSS, JS, sounds
```

## Configuration

Edit `config.py` to adjust:
- **Detection thresholds**: `VIOLENCE_THRESHOLD`, `POSE_MODEL_COMPLEXITY`
- **Performance**: `VIDEO_FRAME_SKIP`, `MODEL_COMPLEXITY`
- **Alerts**: `ALERT_COOLDOWN_SECONDS`, `SMTP_ENABLED`
- **Paths**: Model locations, log file, database

## Dataset Requirements

The violence detection model requires training data:
- **RWF-2000**: 2000 clips from surveillance footage (https://github.com/mchasham/RWF-2000)
- **Hockey Fight**: 1000 NHL fight scenes (available on Kaggle)
- **Custom**: Collect scenario-specific data with `collect_pose_data.py`

## Performance Targets

| Metric | CPU | GPU |
|--------|-----|-----|
| FPS | >= 15 | >= 25 |
| Alert Latency | < 2s | < 1s |
| Violence Accuracy (RWF-2000) | 85%+ | 90%+ |

## Environment Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Email Alerts Setup

1. Enable "Less secure app access" or use App Password (Gmail)
2. Copy `.env.example` to `.env` and fill credentials
3. Set `SMTP_ENABLED = True` in `config.py`

## Troubleshooting

- **Camera not found**: Try different camera index: `python demo.py --camera 1`
- **Low FPS**: Increase frame skip: set `VIDEO_FRAME_SKIP = 3` in `config.py`
- **MediaPipe error**: Ensure `opencv-python` and `mediapipe` are installed
- **TensorFlow GPU**: Install `tensorflow-gpu` package and CUDA for GPU acceleration

## License

MIT License — Open for academic and commercial use.
