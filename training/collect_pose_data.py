"""
Data Collection Script — Record pose keypoint sequences for action classification.

Usage:
    python training/collect_pose_data.py --action standing
    python training/collect_pose_data.py --action walking

Captures 30-frame sequences to datasets/pose_sequences/{action}/
"""

import os
import sys
import time
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.pose_estimator import PoseEstimator
from src.utils import setup_logger, landmarks_to_array

logger = setup_logger("DataCollector")

ACTIONS = ["standing", "walking", "sitting_down", "standing_up",
           "raising_hand", "jumping", "waving", "pointing"]
WINDOW = 30  # frames per sequence


def collect(action, camera=0, max_sequences=100):
    """Live collection: press SPACE to save a sequence, Q to quit."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "datasets", "pose_sequences", action)
    os.makedirs(data_dir, exist_ok=True)

    pose = PoseEstimator()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera}")
        return

    frame_buffer = []
    seq_count = 0
    logger.info(f"Collecting '{action}' data. SPACE=capture, R=reset, Q=quit")

    while seq_count < max_sequences:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, _ = pose.process_frame(frame)
        kp = landmarks_to_array(landmarks)
        frame_buffer.append(kp)

        # Show with skeleton
        display = frame.copy()
        h, w = frame.shape[:2]
        if landmarks is not None:
            from src.utils import draw_skeleton
            draw_skeleton(display, landmarks, w, h)

        # Buffer indicator
        fill = min(len(frame_buffer), WINDOW)
        cv2.putText(display, f"Action: {action} | Buffer: {fill}/{WINDOW} | "
                    f"Seqs saved: {seq_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Highlight when ready to save
        if len(frame_buffer) >= WINDOW:
            cv2.rectangle(display, (10, h - 50), (300, h - 10), (0, 255, 0), -1)
            cv2.putText(display, "Press SPACE to save sequence", (20, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Data Collection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and len(frame_buffer) >= WINDOW:
            seq = np.array(frame_buffer[-WINDOW:])
            fname = os.path.join(data_dir, f"seq_{seq_count:04d}.npy")
            np.save(fname, seq)
            seq_count += 1
            frame_buffer = []
            logger.info(f"Saved sequence {seq_count} to {fname}")
        elif key == ord('r'):
            frame_buffer = []

    logger.info(f"Collected {seq_count} sequences for '{action}'")
    cap.release()
    pose.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, required=True, choices=ACTIONS)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max", type=int, default=100)
    args = parser.parse_args()
    collect(args.action, camera=args.camera, max_sequences=args.max)
