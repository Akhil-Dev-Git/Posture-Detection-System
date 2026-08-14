import tensorflow as tf
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
    for gpu in gpus:
        print(f" - {gpu}")
else:
    print("No GPU detected by TensorFlow.")

try:
    import cv2
    print("OpenCV Version:", cv2.__version__)
except ImportError:
    print("OpenCV not found.")
