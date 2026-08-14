"""Tests for action_recognizer and violence_detector."""
import pytest
import numpy as np

from src.action_recognizer import ActionRecognizer
from src.violence_detector import ViolenceDetector


class TestActionRecognizer:
    def test_init(self):
        ar = ActionRecognizer()
        assert len(ar.buffer) == 0
        assert ar.model is None

    def test_add_frame(self):
        ar = ActionRecognizer()
        # Create synthetic landmark list matching MediaPipe format
        class FakeLM:
            def __init__(self):
                self.x = 0.5
                self.y = 0.5
                self.z = 0.0
                self.visibility = 1.0
        class FakeLandmarks:
            landamrk = [FakeLM() for _ in range(33)]  # Note: matches iteration
            def __iter__(self):
                return iter([FakeLM() for _ in range(33)])
        ar.add_frame(FakeLandmarks())
        assert len(ar.buffer) == 1

    def test_predict_empty(self):
        ar = ActionRecognizer()
        label, conf = ar.predict()
        assert label == "unknown"
        assert conf == 0.0

    def test_heuristic_fallback(self):
        ar = ActionRecognizer()
        seq = np.random.randn(30, 132) * 0.001  # Low motion
        label, conf = ar._heuristic(seq)
        assert label in ["unknown", "standing", "sitting_down"]  # All are low-movement
