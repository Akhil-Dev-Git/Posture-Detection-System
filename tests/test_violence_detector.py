"""Tests for violence detector and alert_manager."""
import os
import pytest
import numpy as np
import tempfile

from src.violence_detector import ViolenceDetector


class TestViolenceDetector:
    def test_init_no_model(self):
        vd = ViolenceDetector()
        assert vd.model is None  # No model file exists
        assert vd.is_violent is False
        assert vd.last_violence_prob == 0.0

    def test_add_frame(self):
        vd = ViolenceDetector()
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        vd.add_frame(frame)
        assert len(vd.frame_buffer) == 1

    def test_buffer_overflow(self):
        vd = ViolenceDetector()
        for _ in range(20):
            vd.add_frame(np.zeros((224, 224, 3), dtype=np.uint8))
        assert len(vd.frame_buffer) <= 16

    def test_heuristic_no_motion(self):
        vd = ViolenceDetector()
        for _ in range(10):
            vd.add_frame(np.zeros((224, 224, 3), dtype=np.float32))
        is_violent, prob = vd.predict()
        assert prob < 0.3  # Should be low with no motion

    def test_reset(self):
        vd = ViolenceDetector()
        vd.add_frame(np.zeros((224, 224, 3), dtype=np.uint8))
        vd.reset()
        assert len(vd.frame_buffer) == 0
        assert vd.frame_counter == 0
