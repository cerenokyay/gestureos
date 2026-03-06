# GestureOS ✋💻

GestureOS is a computer vision project that allows users to **control their computer using hand gestures** through a webcam.

The system detects hand landmarks in real time and maps recognized gestures to actions such as **Enter, Escape, Tab, or custom keyboard shortcuts**.

---

# Features

- Real-time webcam hand tracking
- Hand landmark detection using MediaPipe
- Rule-based gesture recognition (MVP)
- Gesture → action mapping via configuration file
- Cooldown system to prevent repeated triggers
- Modular and extensible architecture

---

# Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI

---

# Example Gestures

| Gesture | Meaning | Action |
|-------|--------|--------|
| OPEN_PALM | Confirm / OK | Enter |
| FIST | Cancel | Escape |
| POINT | Next | Tab |
| THUMBS_UP | Close Tab | Ctrl + W |

Gesture mappings can be changed in:
