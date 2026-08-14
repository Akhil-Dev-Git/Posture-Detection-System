"""
Configuration constants for the Posture Detection System.

All tunable parameters live here for easy competition tuning.
"""

import os

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Pose Estimation
POSE_MODEL_COMPLEXITY = 2  # 0=lite, 1=full, 2=heavy
POSE_SMOOTH_LANDMARKS = True
POSE_ENABLE_SEGMENTATION = False

# === Video Processing ===
VIDEO_FRAME_SKIP = 2        # Process every Nth frame (1=every frame)
VIDEO_SCALE_FACTOR = 1.0    # 1.0 = original, 0.5 = half resolution
DISPLAY_SCALE = 1.0
MAX_FPS = 30

# Action Recognition
ACTION_WINDOW_SIZE = 30               # Number of frames for action sequence
ACTION_KEYPOINT_DIM = 132             # 33 keypoints * 4 (x, y, z, visibility)
ACTION_MODEL_PATH = os.path.join(MODEL_DIR, "action_recognition_lstm.h5")

# Violence Detection
VIOLENCE_INPUT_FRAMES = 16            # Frame buffer for violence model
VIOLENCE_FRAME_RESOLUTION = 224       # MobileNetV2 input size
VIOLENCE_MODEL_PATH = os.path.join(MODEL_DIR, "violence_detection_pytorch.pth")
VIOLENCE_THRESHOLD = 0.3              # Probability threshold to trigger (lowered for demo)
VIOLENCE_CONSECUTIVE_DETECTIONS = 2   # Temporal smoothing: N consecutive positives
VIOLENCE_INFERENCE_INTERVAL = 5       # Run model every N frames

# Posture Classification
POSTURE_CONFIDENCE_THRESHOLD = 0.7    # Below this, use ML fallback
POSTURE_MODEL_PATH = os.path.join(MODEL_DIR, "posture_classifier.pkl")
JOINT_ANGLE_TOLERANCE = 5.0           # Degrees tolerance for rule matching

# Rule-based posture thresholds
STANDING_MIN_ANGLE = 160.0   # Hip-knee-ankle angle
SITTING_HIP_KNEE_MIN = 70.0
SITTING_HIP_KNEE_MAX = 120.0
BENDING_MAX_TORSO_ANGLE = 60.0
RAISE_HANDS_MIN_BEND = 120.0  # Elbow angle

# Alert System
ALERT_COOLDOWN_SECONDS = 30
ALERT_SOUND_PATH = os.path.join(STATIC_DIR, "sounds", "alert.mp3")
ALERT_LOG_PATH = os.path.join(LOG_DIR, "alerts.log")

# Email alerts (SMTP) — fill for competition demo
SMTP_ENABLED = False
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = ""
SMTP_PASSWORD = ""
SMTP_RECIPIENT = ""

# SMS alerts (Twilio) — optional
SMS_ENABLED = False
TWILIO_SID = ""
TWILIO_TOKEN = ""
TWILIO_PHONE = ""
ALERT_PHONE = ""

# Flask
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5050
FLASK_DEBUG = False

# Database
DATABASE_PATH = os.path.join(BASE_DIR, "posture_system.db")
