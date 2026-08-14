"""Tests for pose estimator module."""
import pytest
import numpy as np


class TestPoseEstimator:
    def test_import(self):
        from src.pose_estimator import PoseEstimator
        assert PoseEstimator is not None

    def test_instantiation(self):
        from src.pose_estimator import PoseEstimator
        est = PoseEstimator(model_complexity=0)
        # New mediapipe uses _pose_model, old uses .pose
        assert est._pose_model is not None or est.pose is not None
        est.close()

    def test_close(self):
        from src.pose_estimator import PoseEstimator
        est = PoseEstimator()
        est.close()  # Should not raise

    def test_process_synthetic_frame(self):
        from src.pose_estimator import PoseEstimator
        est = PoseEstimator(model_complexity=0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks, world_landmarks = est.process_frame(frame)
        # Should return None for no-person frame (no exception)
        est.close()
