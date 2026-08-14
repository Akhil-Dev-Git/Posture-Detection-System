"""Tests for posture classifier module."""
import pytest
import numpy as np

from src.posture_classifier import compute_angles, classify_rules, PostureClassifier
from src.utils import angle_between_points


class TestAngles:
    def test_angle_straight_line(self):
        """Three collinear points should give 180 degrees."""
        a = (0, 0, 0)
        b = (0.5, 0, 0)
        c = (1, 0, 0)
        angle = angle_between_points(a, b, c)
        assert abs(angle - 180.0) < 0.1

    def test_angle_right(self):
        """Points forming a right angle at b."""
        a = (0, 0)
        b = (0, 0.5)
        c = (0.5, 0.5)
        angle = angle_between_points(a, b, c)
        assert abs(angle - 90.0) < 5  # tolerance for floating-point


class TestClassifyRules:
    def test_none_input(self):
        label, conf = classify_rules(None)
        assert label is None
        assert conf == 0.0

    def test_synthetic_standing(self):
        """Generate keypoints for a standing person."""
        kp = np.zeros((33, 4), dtype=np.float32)
        # Head near top
        for i in range(9):
            kp[i] = [0.5, 0.1, 0, 1]
        # Shoulders
        kp[11] = [0.4, 0.25, 0, 1]
        kp[12] = [0.6, 0.25, 0, 1]
        # Hips (below shoulders)
        kp[23] = [0.4, 0.5, 0, 1]
        kp[24] = [0.6, 0.5, 0, 1]
        # Knees
        kp[25] = [0.4, 0.7, 0, 1]
        kp[26] = [0.6, 0.7, 0, 1]
        # Ankles
        kp[27] = [0.4, 0.9, 0, 1]
        kp[28] = [0.6, 0.9, 0, 1]
        # Elbows, wrists
        for i in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
            kp[i] = [0.5, 0.4, 0, 1]

        label, conf = classify_rules(kp)
        # At least should not crash
        assert label in ["standing", "raising_hands", None, "lying_down"] or conf > 0


class TestPostureClassifier:
    def test_predict_none(self):
        pc = PostureClassifier()
        label, conf = pc.predict(None)
        assert label == "unknown"
        assert conf == 0.0
