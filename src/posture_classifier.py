"""
Posture Classifier — Hybrid rule-based + ML posture detection.

Uses geometric joint-angle thresholds as the primary classification path,
with an ML fallback (Random Forest) for ambiguous cases.

Postures: standing, sitting, bending, raising_hands, jumping, lying_down
"""

import os
import joblib
import numpy as np
from collections import deque, Counter

import config
from src.utils import (
    setup_logger, landmarks_to_array, angle_between_points, draw_label
)

logger = setup_logger("PostureClassifier")


# === Joint angle computation ===
def compute_angles(kp):
    """
    Compute a dict of clinically-relevant body angles from 33 keypoints.
    kp: np.ndarray (33, 4)
    """
    angles = {}
    # Left arm angles
    angles["l_elbow"] = angle_between_points(kp[11][:2], kp[13][:2], kp[15][:2])
    # Right arm angles
    angles["r_elbow"] = angle_between_points(kp[12][:2], kp[14][:2], kp[16][:2])
    # Left leg angles
    angles["l_hip"] = angle_between_points(kp[23][:2], kp[11][:2], kp[12][:2])
    angles["l_knee"] = angle_between_points(kp[23][:2], kp[25][:2], kp[27][:2])
    # Right leg angles
    angles["r_hip"] = angle_between_points(kp[24][:2], kp[12][:2], kp[11][:2])
    angles["r_knee"] = angle_between_points(kp[24][:2], kp[26][:2], kp[28][:2])
    # Spine / torso
    angles["torso"] = angle_between_points(
        np.array([0.5, 0.0]),  # vertical reference
        (kp[11][:2] + kp[12][:2]) / 2,
        (kp[23][:2] + kp[24][:2]) / 2,
    )
    # Wrist-to-shoulder vertical check
    angles["l_wrist_above_shoulder"] = bool(kp[15][1] < kp[11][1])
    angles["r_wrist_above_shoulder"] = bool(kp[16][1] < kp[12][1])
    return angles


# === Rule-based classification ===
def classify_rules(kp):
    """
    Apply geometric thresholds to classify posture.
    Returns (label, confidence) or (None, 0.0) if no rule matches.
    """
    if kp is None:
        return None, 0.0

    angles = compute_angles(kp)
    l_knee = angles.get("l_knee", 180)
    r_knee = angles.get("r_knee", 180)
    l_elbow = angles.get("l_elbow", 180)
    r_elbow = angles.get("r_elbow", 180)
    torso = angles.get("torso", 0)
    tol = config.JOINT_ANGLE_TOLERANCE

    # --- Lying Down: shoulder & hip Y-levels are similar, body horizontal ---
    shoulder_h = abs(kp[11][1] - kp[12][1])
    hip_h = abs(kp[23][1] - kp[24][1])
    avg_shoulders = (kp[11][1] + kp[12][1]) / 2
    avg_hips = (kp[23][1] + kp[24][1]) / 2
    shoulder_hip_dy = abs(avg_shoulders - avg_hips)
    hip_ankle_dy = abs((kp[23][1] + kp[24][1])/2 - (kp[27][1] + kp[28][1])/2)

    if shoulder_hip_dy < 0.15 and abs(kp[11][:2][0] - kp[27][:2][0]) > 0.15:
        # body horizontal spread
        return "lying_down", 0.9

    # --- Jumping: ankles above hip level ---
    avg_ankle_y = (kp[27][1] + kp[28][1]) / 2
    avg_hip_y = (kp[23][1] + kp[24][1]) / 2
    if avg_ankle_y < avg_hip_y - 0.02:
        return "jumping", 0.85

    # --- Sitting: hip-knee folded 70-120, hips ~ knee height ---
    if (config.SITTING_HIP_KNEE_MIN - tol <= l_knee <= config.SITTING_HIP_KNEE_MAX + tol and
        abs(kp[23][1] - kp[27][1]) < 0.15):
        return "sitting", 0.85

    # --- Bending: torso angle < 60 from vertical ---
    if torso < config.BENDING_MAX_TORSO_ANGLE + tol and kp[11][1] > kp[0][1]:
        return "bending", 0.8

    # --- Raising hands ---
    l_up = angles.get("l_wrist_above_shoulder", False)
    r_up = angles.get("r_wrist_above_shoulder", False)
    if l_up and r_up:
        conf = 0.9
    elif l_up or r_up:
        conf = 0.75
    else:
        conf = 0.0
    if conf >= 0.6:
        return "raising_hands", conf

    # --- Standing: hips above knees, torso upright ---
    l_ankle = angles.get("l_knee", 180)
    r_ankle = angles.get("r_knee", 180)
    hips_above = kp[23][1] < kp[27][1] and kp[24][1] < kp[28][1]
    if l_knee > config.STANDING_MIN_ANGLE - tol and hips_above:
        return "standing", 0.85

    return None, 0.0


# === ML Fallback ===
class PostureClassifier:
    """
    Primary interface for posture classification.
    Uses rule-based first, falls back to trained Random Forest on low-confidence.
    Maintains a frame history for temporal smoothing.
    """

    POSTURES = ["standing", "sitting", "raising_hands", "jumping", "bending", "lying_down"]

    def __init__(self, model_path=None):
        self.model = None
        model_path = model_path or config.POSTURE_MODEL_PATH
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Loaded posture ML model from {model_path}")
            except Exception:
                logger.info("No posture ML model available, using rules only")
        else:
            logger.info("No posture ML model file found, using rules only")
        self.history = deque(maxlen=5)

    def predict(self, kp_array):
        """
        Predict posture for a single frame's keypoints.
        Returns (posture_label, confidence).
        """
        if kp_array is None:
            return "unknown", 0.0
        # Rule-based path
        label, conf = classify_rules(kp_array)
        if label is not None:
            self.history.append(label)
            return label, conf

        # ML fallback
        if self.model is not None:
            features = kp_array.flatten()
            features = features / (np.linalg.norm(features) + 1e-8)
            try:
                probs = self.model.predict_proba(features.reshape(1, -1))[0]
                idx = np.argmax(probs)
                label = self.POSTURES[idx]
                conf = float(probs[idx])
                self.history.append(label)
                return label, conf
            except Exception:
                pass

        self.history.append("unknown")
        return "unknown", 0.0

    def smoothed_predict(self, kp_array):
        """Predict with majority vote smoothing over recent history."""
        label, conf = self.predict(kp_array)
        counts = Counter(self.history)
        label = counts.most_common(1)[0][0]
        return label, conf

    def draw(self, frame, posture, conf):
        color = (0, 200, 0) if conf > 0.8 else (200, 200, 0)
        return draw_label(frame, f"Posture: {posture.upper()}",
                          bg_color=(0, 0, 0), text_color=color, y_offset=0)
