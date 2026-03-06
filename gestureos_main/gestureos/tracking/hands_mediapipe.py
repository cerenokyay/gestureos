import cv2
import numpy as np
import mediapipe as mp


class HandTracker:
    def __init__(self, max_hands: int = 1, det_conf: float = 0.6, track_conf: float = 0.6):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )

    def process(self, frame_bgr):
        annotated = frame_bgr.copy()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if not res.multi_hand_landmarks:
            return None, annotated

        hand = res.multi_hand_landmarks[0]
        self.mp_draw.draw_landmarks(annotated, hand, self.mp_hands.HAND_CONNECTIONS)
        lm = np.array([[p.x, p.y] for p in hand.landmark], dtype=np.float32)
        return lm, annotated

    def close(self):
        self.hands.close()