"""
Alert Manager — Multi-channel alert generation and dispatch.

Handles visual, audio, email, and logging alerts when violence is detected.
Implements cooldown logic to prevent alert flooding.
"""

import os
import time
import smtplib
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import config
from src.utils import setup_logger

logger = setup_logger("AlertManager")
# Separate file logger for alerts
file_handler = logging.FileHandler(config.ALERT_LOG_PATH)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
file_handler.setLevel(logging.INFO)


class AlertManager:
    """
    Manages all outgoing alerts for the system.
    Cooldown prevents spam when violence is continuously detected.
    """

    def __init__(self):
        self.cooldown_seconds = config.ALERT_COOLDOWN_SECONDS
        self.last_alert_time = 0.0
        self.alert_history = []
        self.alert_count = 0
        os.makedirs(config.LOG_DIR, exist_ok=True)

    def handle_detection(self, probability, frame=None):
        """
        Evaluate whether to fire an alert based on violence probability.
        Returns dict of alert info or None if no alert fired.
        """
        if probability < config.VIOLENCE_THRESHOLD:
            return None

        now = time.time()
        if now - self.last_alert_time < self.cooldown_seconds:
            logger.info(f"Alert suppressed (cooldown: {self.cooldown_seconds}s)")
            return None

        self.last_alert_time = now
        self.alert_count += 1
        alert = {
            "id": self.alert_count,
            "timestamp": datetime.now().isoformat(),
            "type": "violence",
            "probability": round(probability, 3),
            "status": "unacknowledged",
        }

        # Fire all active channels
        self._log_alert(alert)
        self._trigger_audio()
        self._send_email(alert, frame)
        self.alert_history.append(alert)

        logger.warning(f"*** ALERT #{self.alert_count}: Violence detected "
                       f"(confidence: {probability:.2%}) ***")
        return alert

    def _log_alert(self, alert):
        """Write alert to log file."""
        with file_handler as _:
            logger.info(f"ALERT: {alert['timestamp']} | {alert['type']} | "
                        f"confidence={alert['probability']}")

    def _trigger_audio(self):
        """Play alert sound."""
        sound_paths = [
            config.ALERT_SOUND_PATH,
            os.path.join(config.STATIC_DIR, "sounds", "alert.wav"),
            os.path.join(config.STATIC_DIR, "sounds", "alert.mp3"),
        ]
        for sound_path in sound_paths:
            if os.path.exists(sound_path):
                try:
                    import playsound
                    playsound.playsound(sound_path, block=False)
                    return
                except ImportError:
                    pass
                except Exception:
                    pass

        # Windows beep fallback
        try:
            import winsound
            winsound.Beep(1000, 400)
            winsound.Beep(1200, 400)
            winsound.Beep(1000, 400)
        except ImportError:
            pass

    def _send_email(self, alert, frame=None):
        """Send email notification with alert details."""
        if not config.SMTP_ENABLED:
            return
        try:
            msg = MIMEMultipart()
            msg["From"] = config.SMTP_USER
            msg["To"] = config.SMTP_RECIPIENT
            msg["Subject"] = f"ALERT: Violence Detected ({alert['timestamp']})"

            body = (
                f"Alert Type: {alert['type']}\n"
                f"Timestamp: {alert['timestamp']}\n"
                f"Confidence: {alert['probability']:.2%}\n"
                f"Alert ID: {alert['id']}"
            )
            msg.attach(MIMEText(body, "plain"))

            if frame is not None:
                import cv2
                tmp_path = os.path.join(config.LOG_DIR,
                                        f"alert_{alert['id']}.jpg")
                cv2.imwrite(tmp_path, frame)
                with open(tmp_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition",
                                    f"attachment; filename=screenshot.jpg")
                    msg.attach(part)

            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
                server.starttls()
                server.login(config.SMTP_USER, config.SMTP_PASSWORD)
                server.send_message(msg)
            logger.info(f"Alert email sent for alert #{alert['id']}")
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")

    def get_history(self, limit=50):
        """Return recent alert history for dashboard display."""
        return self.alert_history[-limit:]

    def acknowledge_alert(self, alert_id):
        """Mark an alert as acknowledged."""
        for alert in self.alert_history:
            if alert["id"] == alert_id:
                alert["status"] = "acknowledged"
                return True
        return False

    def get_stats(self):
        """Return alert statistics."""
        total = len(self.alert_history)
        acknowledged = sum(1 for a in self.alert_history if a["status"] == "acknowledged")
        return {
            "total_alerts": total,
            "acknowledged": acknowledged,
            "unacknowledged": total - acknowledged,
        }
