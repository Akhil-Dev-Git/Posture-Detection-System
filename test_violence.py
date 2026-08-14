import numpy as np
import cv2
import config
from src.violence_detector import ViolenceDetector
import sys
import logging

logging.basicConfig(level=logging.INFO)

print("Starting ViolenceDetector test...")
detector = ViolenceDetector()

print("Model loaded?", detector.model is not None)
print("Is PyTorch?", getattr(detector, 'is_pytorch', False))

# Create dummy frames
print("Generating dummy frames...")
for i in range(16):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detector.add_frame(frame)

print("Running predict...")
try:
    is_violent, prob = detector.predict()
    print("Prediction successful:", is_violent, prob)
except Exception as e:
    print("FATAL ERROR during predict:")
    import traceback
    traceback.print_exc()

sys.exit(0)
