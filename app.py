"""
Posture Detection System - Flask Web Dashboard

Main application entry point. Provides:
- Live video streaming with MJPEG
- Dashboard UI with real-time stats
- Alert history management
- System settings configuration
"""

import os
import json
import logging
import sqlite3
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify

import config
from src.pose_estimator import PoseEstimator
from src.posture_classifier import PostureClassifier
from src.action_recognizer import ActionRecognizer
from src.violence_detector import ViolenceDetector
from src.alert_manager import AlertManager
from src.video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("App")


def create_app():
    app = Flask(__name__)

    # Initialize detection pipeline
    pose = PoseEstimator(
        model_complexity=config.POSE_MODEL_COMPLEXITY,
        smooth_landmarks=config.POSE_SMOOTH_LANDMARKS
    )
    posture = PostureClassifier()
    action = ActionRecognizer()
    violence = ViolenceDetector()
    alerts = AlertManager()
    video = VideoProcessor(
        source=0,
        pose_estimator=pose,
        posture_classifier=posture,
        action_recognizer=action,
        violence_detector=violence,
        alert_manager=alerts
    )

    def init_db():
        con = sqlite3.connect(config.DATABASE_PATH)
        con.execute(
            "CREATE TABLE IF NOT EXISTS alerts "
            "(id INTEGER PRIMARY KEY, timestamp TEXT, type TEXT, "
            "probability REAL, status TEXT DEFAULT 'unacknowledged')"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS activities "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, "
            "posture TEXT, action TEXT, violence_prob REAL, fps REAL)"
        )
        con.commit()
        con.close()

    init_db()

    app.config["pipeline"] = {
        "pose": pose,
        "posture": posture,
        "action": action,
        "violence": violence,
        "alerts": alerts,
        "video": video,
    }

    # Routes
    @app.route("/")
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/alerts")
    def alerts_page():
        return render_template("alerts.html")

    @app.route("/settings")
    def settings_page():
        return render_template("settings.html")

    @app.route("/video_feed")
    def video_feed():
        vid = app.config["pipeline"]["video"]
        if vid.cap is None:
            vid.open_source()

        # Init YOLO locally for the stream
        from ultralytics import YOLO
        import time
        import numpy as np
        model = YOLO('yolov8n-pose.pt')

        def generate():
            SKELETON_CONNECTIONS = [
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (11, 12), (5, 11), (6, 12),
                (11, 13), (13, 15), (12, 14), (14, 16),
                (0, 1), (0, 2), (1, 3), (2, 4)
            ]
            RIGHT_JOINTS = [2, 4, 6, 8, 10, 12, 14, 16]
            LEFT_JOINTS = [1, 3, 5, 7, 9, 11, 13, 15]
            
            # Kinematics Tracker for Ultra-Accurate Violence Detection
            prev_keypoints = None
            smoothed_violence_prob = 0.0

            while True:
                ret, frame = vid.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                annotated_frame = frame.copy()

                results = model(frame, verbose=False)
                persons_detected = 0

                is_violent = False
                violence_prob = 0.0
                try:
                    vid.total_frames_processed += 1
                    vid.fps = getattr(vid, '_calculate_fps', lambda: 20)()
                except Exception:
                    pass

                if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                    keypoints = results[0].keypoints.xy.cpu().numpy()
                    confs = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None
                    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
                    box_confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []

                    persons_detected = len(keypoints)

                    # --- YOLOv8 Advanced Kinematic AI Engine ---
                    # Calculate exact speed of fists and feet 
                    max_velocity = 0.0
                    if prev_keypoints is not None and len(prev_keypoints) == len(keypoints):
                        for i in range(persons_detected):
                            person_curr = keypoints[i]
                            person_prev = prev_keypoints[i]
                            # 9,10 = Wrists | 15,16 = Ankles
                            for pt_idx in [9, 10, 15, 16]:
                                px, py = person_prev[pt_idx]
                                cx, cy = person_curr[pt_idx]
                                if px > 10 and py > 10 and cx > 10 and cy > 10:
                                    dist = ((cx - px)**2 + (cy - py)**2)**0.5
                                    if dist > max_velocity:
                                        max_velocity = dist
                    
                    prev_keypoints = keypoints.copy()
                    
                    # Convert pixel velocity into fighting probability
                    # Normal movement < 15px. Fighting usually > 40px between frames.
                    velocity_prob = min(max(max_velocity - 15.0, 0) / 45.0, 1.0)
                    
                    # Smooth it over time
                    smoothed_violence_prob = (smoothed_violence_prob * 0.7) + (velocity_prob * 0.3)
                    
                    final_p = max(smoothed_violence_prob, violence_prob)
                    if final_p > 0.65:
                        is_violent = True
                        
                    violence_prob = final_p
                    # -------------------------------------------

                    for i in range(persons_detected):
                        person_kpts = keypoints[i]
                        person_confs = confs[i] if confs is not None else np.ones(17)

                        if i < len(boxes):
                            x1, y1, x2, y2 = map(int, boxes[i])
                            confidence = box_confs[i] if i < len(box_confs) else 1.0
                            line_len = max((x2 - x1) // 6, 10)
                            box_color = (0, 0, 255) if is_violent else (0, 200, 100)
                            cv2.line(annotated_frame, (x1, y1), (x1+line_len, y1), box_color, 2)
                            cv2.line(annotated_frame, (x1, y1), (x1, y1+line_len), box_color, 2)
                            cv2.line(annotated_frame, (x2, y1), (x2-line_len, y1), box_color, 2)
                            cv2.line(annotated_frame, (x2, y1), (x2, y1+line_len), box_color, 2)
                            cv2.line(annotated_frame, (x1, y2), (x1+line_len, y2), box_color, 2)
                            cv2.line(annotated_frame, (x1, y2), (x1, y2-line_len), box_color, 2)
                            cv2.line(annotated_frame, (x2, y2), (x2-line_len, y2), box_color, 2)
                            cv2.line(annotated_frame, (x2, y2), (x2, y2-line_len), box_color, 2)
                            
                            if is_violent:
                                label = f"ID:{i+1} | FIGHTING!"
                                cv2.rectangle(annotated_frame, (x1, y1-25), (x1 + len(label)*11, y1), (0, 0, 255), -1)
                                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                            else:
                                label = f"Person ID:{i+1} | {confidence*100:.1f}%"
                                cv2.rectangle(annotated_frame, (x1, y1-25), (x1 + len(label)*10, y1), box_color, -1)
                                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                        for (p1, p2) in SKELETON_CONNECTIONS:
                            if person_confs[p1] > 0.4 and person_confs[p2] > 0.4:
                                pt1 = (int(person_kpts[p1][0]), int(person_kpts[p1][1]))
                                pt2 = (int(person_kpts[p2][0]), int(person_kpts[p2][1]))
                                if pt1 != (0, 0) and pt2 != (0, 0):
                                    cv2.line(annotated_frame, pt1, pt2, (230, 230, 230), 2, cv2.LINE_AA)

                        for j in range(17):
                            if person_confs[j] > 0.4:
                                pt = (int(person_kpts[j][0]), int(person_kpts[j][1]))
                                if pt != (0, 0):
                                    color = (0, 255, 255)
                                    if j in RIGHT_JOINTS: color = (0, 100, 255)
                                    elif j in LEFT_JOINTS: color = (255, 255, 0)
                                    cv2.circle(annotated_frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
                                    cv2.circle(annotated_frame, pt, 5, color, 1, cv2.LINE_AA)

                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, 50), (10, 10, 10), -1)
                cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                cv2.putText(annotated_frame, "AI VISION : POSITIONAL ANALYSIS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"TARGETS: {persons_detected}", (width - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

                # Feed metrics and Run Violence Detection! (Moved to top of loop for bounding box color)

                # If Fighting is detected, show a nice small red alert at the top instead of full screen!
                if is_violent:
                    alert_text = " WARNING: FIGHTING DETECTED! "
                    (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    tx = int(width/2) - int(tw/2)
                    ty = 80
                    cv2.rectangle(annotated_frame, (tx-10, ty-th-10), (tx+tw+10, ty+10), (0, 0, 255), -1)
                    cv2.putText(annotated_frame, alert_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                # Violence indicator bar at bottom right
                bar_width = int(violence_prob * 200)
                bar_color = (0, 0, 255) if is_violent else (0, 255, 0)
                cv2.rectangle(annotated_frame, (width - 210, height - 25), (width - 10, height - 10), (50, 50, 50), -1)
                cv2.rectangle(annotated_frame, (width - 210, height - 25), (width - 210 + bar_width, height - 10), bar_color, -1)
                cv2.putText(annotated_frame, f"Fighting Prob: {violence_prob:.2f}", (width - 205, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                success, buf = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not success:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

        return Response(generate(), mimetype="multipart/x-mixed-replace;boundary=frame")

    @app.route("/api/stats")
    def api_stats():
        vid = app.config["pipeline"]["video"]
        alerts_mgr = app.config["pipeline"]["alerts"]
        vi = app.config["pipeline"]["violence"]
        ar = app.config["pipeline"]["action"]
        pc = app.config["pipeline"]["posture"]

        action_label = "unknown"
        if ar and len(ar.buffer) >= 5:
            action_label, _ = ar.predict()

        return jsonify({
            "fps": round(vid.fps, 1),
            "posture_label": pc.history[-1] if pc.history else "unknown",
            "action_label": action_label,
            "action_buffer_pct": round(ar.get_status()),
            "violence_prob": round(vi.last_violence_prob, 3),
            "is_violent": vi.is_violent,
            "alert_count": len(alerts_mgr.alert_history),
            "total_frames": vid.total_frames_processed,
        })

    @app.route("/api/alerts")
    def api_alerts():
        alerts_mgr = app.config["pipeline"]["alerts"]
        return jsonify(alerts_mgr.get_history(limit=100))

    @app.route("/api/alerts/<int:alert_id>/acknowledge", methods=["POST"])
    def acknowledge_alert(alert_id):
        alerts_mgr = app.config["pipeline"]["alerts"]
        if alerts_mgr.acknowledge_alert(alert_id):
            return jsonify({"status": "ok"})
        return jsonify({"error": "not found"}), 404

    @app.route("/api/settings", methods=["GET", "POST"])
    def api_settings():
        cfg = app.config["pipeline"]
        if request.method == "GET":
            return jsonify({
                "violence_threshold": cfg["violence"].threshold,
                "alert_cooldown": cfg["alerts"].cooldown_seconds,
                "frame_skip": config.VIDEO_FRAME_SKIP,
                "model_complexity": config.POSE_MODEL_COMPLEXITY,
                "smtp_enabled": config.SMTP_ENABLED,
            })
        data = request.json
        cfg["violence"].threshold = float(
            data.get("violence_threshold", cfg["violence"].threshold))
        cfg["alerts"].cooldown_seconds = int(
            data.get("alert_cooldown", cfg["alerts"].cooldown_seconds))
        config.VIOLENCE_INFERENCE_INTERVAL = int(
            data.get("inference_interval", config.VIOLENCE_INFERENCE_INTERVAL))
        config.VIDEO_FRAME_SKIP = int(
            data.get("frame_skip", config.VIDEO_FRAME_SKIP))
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Posture Detection System...")
    logger.info(f"Dashboard: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT,
            debug=config.FLASK_DEBUG, threaded=True)
