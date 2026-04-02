import cv2
import numpy as np
import mediapipe as mp
from collections import deque


class HandTracker:
    def __init__(self, max_hands: int = 1, det_conf: float = 0.6, track_conf: float = 0.6, screen_size: tuple = None):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        
        # Mouse smoothing için (son 5 kare ortalaması)
        self.mouse_history = deque(maxlen=5)
        self.screen_size = screen_size or (1920, 1080)  # Default ekran boyutu

    def process(self, frame_bgr):
        annotated = frame_bgr.copy()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        frame_h, frame_w = frame_bgr.shape[:2]
        
        if not res.multi_hand_landmarks:
            return None, annotated, None

        hand = res.multi_hand_landmarks[0]
        self.mp_draw.draw_landmarks(annotated, hand, self.mp_hands.HAND_CONNECTIONS)
        lm = np.array([[p.x, p.y] for p in hand.landmark], dtype=np.float32)
        
        # Mouse koordinatı hesapla (işaret parmağı ucu - landmark 8)
        index_finger_x = lm[8, 0]  # Normalized (0-1)
        index_finger_y = lm[8, 1]  # Normalized (0-1)
        
        # Ekran koordinatına çevir
        mouse_x = int(index_finger_x * self.screen_size[0])
        mouse_y = int(index_finger_y * self.screen_size[1])
        
        # Smoothing (zıplama önlemek için)
        self.mouse_history.append((mouse_x, mouse_y))
        smoothed_x = int(np.mean([m[0] for m in self.mouse_history]))
        smoothed_y = int(np.mean([m[1] for m in self.mouse_history]))
        
        mouse_pos = (smoothed_x, smoothed_y)
        
        # Debug: Ekranda göster
        cv2.circle(annotated, (int(index_finger_x * frame_w), int(index_finger_y * frame_h)), 5, (0, 255, 255), -1)
        
        return lm, annotated, mouse_pos

    def close(self):
        self.hands.close()
