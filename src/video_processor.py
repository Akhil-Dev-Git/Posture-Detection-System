"""
Video Processor — Frame capture, module orchestration, and annotation.

Coordinates pose estimation, posture classification, action recognition,
and violence detection across video frames with overlay rendering.
"""

import time
import threading
import queue
import cv2
import numpy as np

import config
from src.utils import setup_logger, landmarks_to_array, draw_skeleton, draw_label

logger = setup_logger("VideoProcessor")


class VideoProcessor:
    """
    Central pipeline controller. Manages input source, processes frames,
    coordinates all detection modules, and overlays results.
    """

    def __init__(self, source=0, pose_estimator=None, posture_classifier=None,
                 action_recognizer=None, violence_detector=None, alert_manager=None):
        self.source = source
        self.cap = None
        self.fps = 0
        self.frame_skip = config.VIDEO_FRAME_SKIP
        self.scale_factor = config.VIDEO_SCALE_FACTOR

        # Detection modules
        self.pose_estimator = pose_estimator
        self.posture_classifier = posture_classifier
        self.action_recognizer = action_recognizer
        self.violence_detector = violence_detector
        self.alert_manager = alert_manager



        # MJPEG streaming
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.processing_thread = None
        self.running = False

        # Metrics
        self.prev_time = time.time()
        self.frame_count = 0
        self.total_frames_processed = 0

    def open_source(self):
        """
        Initialize the video source. Source can be:
        - int (webcam index)
        - string (RTSP URL or video file path)
        """
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened: {width}x{height} @ {fps:.1f} fps")
        return True

    def process_frame(self):
        """
        Read one frame from source, run all detection modules,
        annotate, and overlay results.
        Returns the annotated frame (or None if no frame).
        """
        ret, frame = self.cap.read()
        if not ret:
            return None

        self.fps = self._calculate_fps()
        self.total_frames_processed += 1

        # Scale and mirror for display
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = frame.shape[:2]

        # Skip frames for performance
        if self.frame_count % self.frame_skip != 0:
            return frame

        self.frame_count += 1

        # --- Pose Estimation ---
        landmarks = None
        kp_array = None
        if self.pose_estimator is not None:
            try:
                landmarks, world_landmarks = self.pose_estimator.process_frame(frame)
                if landmarks is not None:
                    kp_array = landmarks_to_array(landmarks)
            except Exception as e:
                logger.warning(f"Pose estimation error: {e}")

        # --- Posture Classification ---
        posture = "unknown"
        posture_conf = 0.0
        if self.posture_classifier is not None and kp_array is not None:
            posture, posture_conf = self.posture_classifier.smoothed_predict(kp_array)

        # --- Action Recognition ---
        action = "unknown"
        action_conf = 0.0
        if self.action_recognizer is not None and landmarks is not None:
            self.action_recognizer.add_frame(landmarks)
            action, action_conf = self.action_recognizer.predict()

        # --- Violence Detection ---
        is_violent = False
        violence_prob = 0.0
        if self.violence_detector is not None:
            self.violence_detector.add_frame(frame)
            is_violent, violence_prob = self.violence_detector.predict()

            # Alert
            if self.alert_manager is not None:
                self.alert_manager.handle_detection(violence_prob, frame=None)

        # --- Draw overlays ---
        display = frame.copy()

        # Human Full Body Bounding Box & Skeleton
        person_box = None
        persons_detected = 0
        if landmarks is not None:
            persons_detected = 1
            draw_skeleton(display, landmarks, w, h)
            
            # Calculate full-body bounding box from landmarks
            x_coords = [lm.x for lm in landmarks.landmark]
            y_coords = [lm.y for lm in landmarks.landmark]
            
            min_x = max(0, int(min(x_coords) * w) - 20)
            max_x = min(w, int(max(x_coords) * w) + 20)
            min_y = max(0, int(min(y_coords) * h) - 40)
            max_y = min(h, int(max(y_coords) * h) + 20)
            
            # Professional Corner Bracket Bounding Box
            line_len = max((max_x - min_x) // 6, 10)
            box_color = (0, 200, 100)
            cv2.line(display, (min_x, min_y), (min_x+line_len, min_y), box_color, 2)
            cv2.line(display, (min_x, min_y), (min_x, min_y+line_len), box_color, 2)
            cv2.line(display, (max_x, min_y), (max_x-line_len, min_y), box_color, 2)
            cv2.line(display, (max_x, min_y), (max_x, min_y+line_len), box_color, 2)
            cv2.line(display, (min_x, max_y), (min_x+line_len, max_y), box_color, 2)
            cv2.line(display, (min_x, max_y), (min_x, max_y-line_len), box_color, 2)
            cv2.line(display, (max_x, max_y), (max_x-line_len, max_y), box_color, 2)
            cv2.line(display, (max_x, max_y), (max_x, max_y-line_len), box_color, 2)
            
            label = "Person ID:1 | 99.9%"
            cv2.rectangle(display, (min_x, min_y-25), (min_x + len(label)*10, min_y), box_color, -1)
            cv2.putText(display, label, (min_x + 5, min_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Professional HUD overlay
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, "AI VISION : POSITIONAL ANALYSIS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f"TARGETS: {persons_detected}", (w - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        # Violence background flash & Marking the Screen
        if is_violent:
            overlay = display.copy()
            # Mark the screen with a red flashing border
            cv2.rectangle(overlay, (10, 10), (w-10, h-10), (0, 0, 255), 15)
            # Add a slight red tint to the whole screen
            cv2.addWeighted(overlay, 0.4,
                            np.full_like(display, (0, 0, 255)), 0.6, 0, display)
            
            # Draw bounding box around the exact fighting location
            if hasattr(self.violence_detector, 'motion_box') and self.violence_detector.motion_box is not None:
                bx, by, bw, bh = self.violence_detector.motion_box
                cv2.rectangle(display, (bx, by), (bx+bw, by+bh), (0, 255, 255), 4)
                cv2.putText(display, "THREAT", (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            display = cv2.putText(
                display, "VIOLENCE DETECTED",
                (int(w/2) - 180, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 255), 4
            )

        # Labels
        if landmarks is not None:
            posture_color = (0, 255, 0) if posture_conf > 0.8 else (200, 200, 0)
            draw_label(display, f"Posture: {posture.upper()}",
                       text_color=posture_color, y_offset=0)
            draw_label(display, f"Action: {action.replace('_', ' ').title()}",
                       text_color=(0, 150, 255), y_offset=30)
            action_buf = ""
            if self.action_recognizer:
                buf_pct = int(self.action_recognizer.get_status())
                action_buf = f" ({buf_pct}%)"
            draw_label(display, f"Action Buffer: {action_buf}",
                       text_color=(150, 150, 255), y_offset=60, font_scale=0.5)

        # Violence indicator bar
        if self.violence_detector is not None:
            bar_width = int(violence_prob * 200)
            bar_color = (0, 0, 255) if violence_prob >= config.VIOLENCE_THRESHOLD else (0, 255, 0)
            cv2.rectangle(display, (w - 210, h - 25), (w - 10, h - 10),
                          (50, 50, 50), -1)
            cv2.rectangle(display, (w - 210, h - 25),
                          (w - 210 + bar_width, h - 10), bar_color, -1)
            cv2.putText(display, f"Violence: {violence_prob:.2f}",
                        (w - 215, h - 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        # FPS counter
        cv2.putText(display, f"FPS: {int(self.fps):02d}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(display, f"Frame: {self.total_frames_processed}", (w - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Update current frame for streaming
        with self.frame_lock:
            self.current_frame = display.copy()

        return display

    def _calculate_fps(self):
        """Calculate rolling FPS."""
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        return (self.fps * 0.95) if dt == 0 else (0.95 * self.fps + 0.05 / dt)

    def start_streaming(self):
        """Start background processing thread for MJPEG streaming."""
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Streaming started")

    def _process_loop(self):
        """Background thread that continuously processes frames."""
        while self.running:
            frame = self.process_frame()
            if frame is None:
                time.sleep(0.01)
                continue

    def get_frame(self):
        """
        Get the latest processed frame (thread-safe),
        encoded as JPEG bytes for MJPEG streaming.
        """
        with self.frame_lock:
            if self.current_frame is None:
                return None
            success, buffer = cv2.imencode('.jpg', self.current_frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not success:
                return None
            return buffer.tobytes()

    def stop(self):
        """Stop the processing thread and release the video capture."""
        self.running = False
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
        if self.pose_estimator is not None:
            self.pose_estimator.close()
        logger.info("Video processor stopped")

    def __del__(self):
        self.stop()
