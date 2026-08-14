"""
Utility functions for the Posture Detection System.
"""

import math
import logging
import os
import sys

import numpy as np
import cv2

# === Logging ===
def setup_logger(name, level=logging.INFO):
    """Configure a module logger with console + file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


# === Geometry helpers ===
def euclidean_distance(p1, p2):
    """Euclidean distance between two 2D/3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def angle_between_points(a, b, c):
    """
    Calculate the absolute angle (degrees) at point b formed by a-b-c.
    a, b, c are (x, y) or (x, y, z) tuples/lists.
    """
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    c_arr = np.array(c, dtype=np.float32)
    ba = a_arr - b_arr
    bc = c_arr - b_arr
    dot = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cos_val = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return math.degrees(math.acos(cos_val))


def is_landmark_visible(landmark, threshold=0.5):
    """Check if a MediaPipe landmark is visible enough to use."""
    val = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
    return val > threshold


# === Keypoint conversion helpers ===
def landmarks_to_array(landmarks):
    """
    Convert MediaPipe landmark list to numpy array [33, 4].
    Returns (x, y, z, visibility) for each keypoint.
    If landmark is missing/not visible, fill with zeros.
    """
    arr = np.zeros((33, 4), dtype=np.float32)
    if landmarks is None:
        return arr
        
    lm_list = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks
    for idx, lm in enumerate(lm_list):
        arr[idx] = [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0]
    return arr


def normalize_keypoints(kp_array):
    """Normalize keypoints to center-of-mass at origin."""
    if kp_array.shape[0] == 0:
        return kp_array
    center = kp_array[:, :2].mean(axis=0)
    kp_array[:, :2] -= center
    return kp_array


# === Drawing helpers ===
def draw_skeleton(frame, landmarks, width, height):
    """Draw MediaPipe skeleton on frame."""
    import mediapipe as mp
    connections = mp.solutions.pose.POSE_CONNECTIONS if hasattr(mp, "solutions") else [
        (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29),
        (27, 31), (24, 26), (26, 28), (28, 30), (28, 32)
    ]
    
    # Professional colors
    RIGHT_JOINTS = {2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32}
    LEFT_JOINTS = {1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31}

    lm_list = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks
    
    # Draw connections (White solid lines)
    for a, b in connections:
        if a >= len(lm_list) or b >= len(lm_list):
            continue
        lm_a = lm_list[a]
        lm_b = lm_list[b]
        if not (is_landmark_visible(lm_a) and is_landmark_visible(lm_b)):
            continue
        x1, y1 = int(lm_a.x * width), int(lm_a.y * height)
        x2, y2 = int(lm_b.x * width), int(lm_b.y * height)
        cv2.line(frame, (x1, y1), (x2, y2), (230, 230, 230), 2, cv2.LINE_AA)
        
    # Draw joints (Color-coded Left=Cyan, Right=Orange, Center=Yellow)
    for idx, lm in enumerate(lm_list):
        if is_landmark_visible(lm):
            x, y = int(lm.x * width), int(lm.y * height)
            
            color = (0, 255, 255) # Yellow for Center (0)
            if idx in RIGHT_JOINTS:
                color = (0, 100, 255) # Orange Right
            elif idx in LEFT_JOINTS:
                color = (255, 255, 0) # Cyan Left

            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA) # White Core
            cv2.circle(frame, (x, y), 5, color, 1, cv2.LINE_AA) # Colored Ring
    
    return frame


def draw_label(frame, text, bg_color=(0, 0, 0), text_color=(255, 255, 255),
               y_offset=0, font_scale=0.7):
    """Draw a semi-transparent label on frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    y = 10 + y_offset
    x = 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w + 8, y + h + 4), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (x + 4, y + h), font, font_scale,
                text_color, thickness)
    return frame
