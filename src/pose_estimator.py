"""
Pose Estimator — Real-time skeletal keypoint extraction using MediaPipe Pose.

Outputs normalized 33-keypoint arrays per frame.
Supports both old solutions API and new task-based API (MediaPipe 0.10+).
"""

import os
import urllib.request
import cv2
import numpy as np

import mediapipe as mp

from src.utils import setup_logger

logger = setup_logger("PoseEstimator")

_USE_SOLUTIONS = hasattr(mp, "solutions")

_MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           ".mediapipe_models")


def _ensure_model(model_name):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fname = os.path.join(_CACHE_DIR, f"pose_{model_name}.task")
    if os.path.exists(fname):
        return fname
    url = _MODEL_URLS.get(model_name, _MODEL_URLS["lite"])
    logger.info(f"Downloading pose model: {model_name} ...")
    with urllib.request.urlopen(url) as resp, open(fname, 'wb') as f:
        f.write(resp.read())
    logger.info(f"Model saved to {fname}")
    return fname


class PoseEstimator:
    def __init__(self, model_complexity=1, smooth_landmarks=True,
                 enable_segmentation=False):
        self.model_complexity = model_complexity

        if _USE_SOLUTIONS:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks,
                enable_segmentation=enable_segmentation,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        else:
            self._pose_model = None
            model_map = {0: "lite", 1: "full", 2: "heavy"}
            model_name = model_map.get(model_complexity, "full")
            self._init_tasks_api(model_name)

    def _init_tasks_api(self, model_name):
        model_path = _ensure_model(model_name)
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._pose_model = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        logger.info(f"Initialized MediaPipe PoseLandmarker ({model_name})")

    def process_frame(self, frame):
        """
        Run pose detection on a single BGR frame.
        Returns (landmark_list, world_landmark_list) or (None, None).
        """
        if _USE_SOLUTIONS:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            return results.pose_landmarks, results.pose_world_landmarks
        else:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                result = self._pose_model.detect(mp_image)
                if result.pose_landmarks:
                    return result.pose_landmarks[0], result.world_landmarks[0]
                return None, None
            except Exception as e:
                logger.warning(f"Pose detection failed: {e}")
                return None, None

    def close(self):
        if _USE_SOLUTIONS:
            self.pose.close()
        elif self._pose_model is not None:
            self._pose_model.close()
