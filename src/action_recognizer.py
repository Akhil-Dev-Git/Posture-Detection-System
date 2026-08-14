"""
Action Recognizer — LSTM-based temporal sequence classification.

Accepts a sliding window of pose keypoint sequences and classifies
the temporal action (walking, waving, sitting_down, etc.).
"""

import os
import numpy as np

import config
from src.utils import setup_logger, landmarks_to_array

logger = setup_logger("ActionRecognizer")

ACTION_LABELS = [
    "standing", "walking", "sitting_down", "standing_up",
    "raising_hand", "jumping", "waving", "pointing"
]


class ActionRecognizer:
    """
    Manages the LSTM model for action recognition.
    Maintains a frame buffer; when full, runs inference.
    Falls back to heuristic-based detection when model not loaded.
    """

    def __init__(self, model_path=None):
        self.window_size = config.ACTION_WINDOW_SIZE
        self.buffer = []
        self.model = None
        model_path = model_path or config.ACTION_MODEL_PATH
        if os.path.exists(model_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                logger.info(f"Loaded action LSTM from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load action model: {e}")
        else:
            logger.info("No action model found — rule-based heuristics active")

    def add_frame(self, landmarks):
        """Add a frame's keypoints to the sliding window buffer."""
        kp = landmarks_to_array(landmarks)
        kp_flat = kp.flatten()  # 33*4 = 132
        self.buffer.append(kp_flat)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]

    def predict(self):
        """Run action classification on the current buffer."""
        if len(self.buffer) < 5:
            return "unknown", 0.0

        if self.model is not None:
            return self._model_predict()
        else:
            return self._heuristic(self.buffer)

    def _model_predict(self):
        """LSTM model inference."""
        try:
            import tensorflow as tf
            seq = np.array(self.buffer[-self.window_size:])
            seq = seq.reshape(1, self.window_size, -1)
            probs = self.model.predict(seq, verbose=0)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            return ACTION_LABELS[idx], conf
        except Exception as e:
            logger.warning(f"Action inference failed: {e}")
            return self._heuristic(self.buffer)

    def _heuristic(self, seq):
        """
        Rule-based action detection from pose keypoint velocity analysis.
        seq: list/list of (132,) numpy arrays
        """
        if len(seq) < 5:
            return "unknown", 0.0

        seq = np.array(seq[-30:])  # use up to last 30 frames
        n = len(seq)

        # --- Feature 1: Wrist velocity (for waving/raising hand) ---
        wrist_idxs = [15, 16]  # keypoint indices for left/right wrist
        wrist_positions = np.array([[s[i*4:i*4+2] for i in wrist_idxs] for s in seq])
        wrist_vel = np.abs(np.diff(wrist_positions, axis=0))
        wrist_vel_x = wrist_vel[:, :, 0].sum(axis=1).mean()  # horizontal movement
        wrist_vel_y = wrist_vel[:, :, 1].sum(axis=1).mean()  # vertical movement
        total_wrist_vel = wrist_vel.sum()

        # --- Feature 2: Ankle velocity (for walking/jumping) ---
        ankle_idxs = [27, 28]  # keypoint indices for ankles
        ankle_positions = np.array([[s[i*4:i*4+2] for i in ankle_idxs] for s in seq])
        ankle_vel = np.abs(np.diff(ankle_positions, axis=0)).sum()

        # --- Feature 3: Hip vertical movement (for jumping) ---
        hip_y = seq[:, 23*4+1]  # left hip y over time
        hip_vel_y = np.abs(np.diff(hip_y)).mean()

        # --- Feature 4: Overall body motion ---
        hip_pos = seq[:, 23*4:23*4+2]
        hip_motion = np.abs(np.diff(hip_pos, axis=0)).sum()

        # --- Feature 5: Wrist above shoulder ---
        wrist_above = []
        for s in seq[-10:]:
            l_wrist_y = s[15*4+1]
            r_wrist_y = s[16*4+1]
            l_shoulder_y = s[11*4+1]
            r_shoulder_y = s[12*4+1]
            wrist_above.append(l_wrist_y < l_shoulder_y or r_wrist_y < r_shoulder_y)
        wrist_above_ratio = sum(wrist_above) / len(wrist_above)

        # --- Decision logic ---
        scores = {
            "standing": 0.0,
            "walking": 0.0,
            "sitting_down": 0.0,
            "standing_up": 0.0,
            "raising_hand": 0.0,
            "jumping": 0.0,
            "waving": 0.0,
            "pointing": 0.0,
        }

        # Jumping: ankle above hip AND hip vertical velocity high
        ankle_y_last = seq[-1, 27*4+1]
        hip_y_last = seq[-1, 23*4+1]
        if ankle_y_last < hip_y_last and hip_vel_y > 0.02:
            scores["jumping"] = 0.85
        elif hip_vel_y > 0.01:
            scores["jumping"] = 0.5

        # Waving: high wrist velocity but low body motion
        if total_wrist_vel > 0.1 and hip_motion < 0.3:
            scores["waving"] = 0.8
        elif total_wrist_vel > 0.05 and hip_motion < 0.2:
            scores["waving"] = 0.6

        # Raising hand: wrists above shoulders
        if wrist_above_ratio > 0.7 and wrist_vel_y > 0.005:
            scores["raising_hand"] = 0.85
        elif wrist_above_ratio > 0.5:
            scores["raising_hand"] = 0.6

        # Walking: moderate ankle motion, moderate hip motion, low wrist
        if ankle_vel > 0.3 and total_wrist_vel < 0.5 and hip_motion > 0.05:
            scores["walking"] = 0.75
        elif ankle_vel > 0.15 and total_wrist_vel < 0.3:
            scores["walking"] = 0.5

        # Standing: low everything
        if total_wrist_vel < 0.02 and ankle_vel < 0.1 and hip_motion < 0.05:
            scores["standing"] = 0.8

        # Sitting down: wrists low, low motion, hip position low
        if hip_motion < 0.05 and total_wrist_vel < 0.05:
            hip_y_norm = seq[-1, 23*4+1]
            if hip_y_norm > 0.4:
                scores["sitting_down"] = 0.5

        # Pointing: one arm extended, one arm still
        if total_wrist_vel < 0.5 and wrist_above_ratio > 0.3:
            scores["pointing"] = 0.4

        # Get highest scoring action
        best = max(scores, key=scores.get)
        conf = scores[best]
        return best, conf

    def get_status(self):
        """Return buffer fill percentage for UI."""
        return min(len(self.buffer) / self.window_size * 100, 100)

    def reset(self):
        self.buffer = []
