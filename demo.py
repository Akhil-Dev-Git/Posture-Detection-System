"""
Standalone Webcam Demo — Runs the detection pipeline in real-time
using OpenCV window (no web dashboard required).
"""

import cv2
import argparse
import time

from src.pose_estimator import PoseEstimator
from src.posture_classifier import PostureClassifier
from src.action_recognizer import ActionRecognizer
from src.violence_detector import ViolenceDetector
from src.utils import setup_logger, landmarks_to_array, draw_skeleton, draw_label

logger = setup_logger("Demo")


def run_demo(camera=0, model_complexity=1):
    pose = PoseEstimator(model_complexity=model_complexity)
    posture = PostureClassifier()
    action = ActionRecognizer()
    violence = ViolenceDetector()

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera {camera} opened: {w}x{h}")

    fps_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pose estimation
        landmarks, _ = pose.process_frame(frame)

        kp_array = None
        posture_label = "unknown"
        posture_conf = 0.0
        action_label = "unknown"
        violence_label = "Safe"

        if landmarks is not None:
            kp_array = landmarks_to_array(landmarks)
            posture_label, posture_conf = posture.smoothed_predict(kp_array)

            action.add_frame(landmarks)
            if len(action.buffer) >= 15:
                action_label, _ = action.predict()

            frame_color = frame

        violence.add_frame(frame)
        is_violent, v_prob = violence.predict()

        # Draw everything
        display = frame.copy()
        if landmarks is not None:
            draw_skeleton(display, landmarks, w, h)

        if is_violent:
            overlay = display.copy()
            cv2.addWeighted(overlay, 0.3,
                            cv2.cvtColor(display, cv2.COLOR_BGR2GRAY),
                            0.7, 0, display)
            cv2.putText(display, "VIOLENCE DETECTED",
                        (int(w/2) - 200, int(h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Labels
        y = 25
        display = draw_label(display, f"Posture: {posture_label.upper()} (conf: {posture_conf:.2f})",
                             y_offset=y)
        y += 25
        buf_pct = int(action.get_status())
        display = draw_label(display, f"Action: {action_label.replace('_', ' ').title()} [{buf_pct}%]",
                             y_offset=y, text_color=(0, 150, 255))
        y += 25
        violence_color = (0, 255, 0)
        if v_prob > 0.4:
            violence_color = (0, 165, 255)
        if v_prob >= violence.threshold:
            violence_color = (0, 0, 255)
        display = draw_label(display, f"Violence: {v_prob:.2%}",
                             y_offset=y, text_color=violence_color)

        # FPS
        frame_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps_val = frame_count / elapsed
            fps_time = time.time()
            frame_count = 0
        else:
            fps_val = 0
        cv2.putText(display, f"FPS: {fps_val:.1f}", (w - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Posture Detection System", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()
    logger.info("Demo ended")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Posture Detection System — Standalone Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                        help="Pose model complexity (0=lite, 1=full, 2=heavy)")
    args = parser.parse_args()
    run_demo(camera=args.camera, model_complexity=args.complexity)
