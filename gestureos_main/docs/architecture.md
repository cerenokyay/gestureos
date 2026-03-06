# Architecture

GestureOS is designed as a modular computer vision pipeline.

Pipeline stages:

1. Camera capture using OpenCV
2. Hand landmark detection using MediaPipe
3. Gesture recognition
4. Action mapping
5. OS input execution

---

# Data Flow

Camera Frame
↓
Hand Landmarks
↓
Gesture Classification
↓
Action Mapping
↓
Keyboard / Mouse Event