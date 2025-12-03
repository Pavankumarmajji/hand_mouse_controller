# check_versions.py
import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy import error: {e}")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"OpenCV import error: {e}")

try:
    import mediapipe as mp
    print(f"MediaPipe version: {mp.__version__}")
except ImportError as e:
    print(f"MediaPipe import error: {e}")