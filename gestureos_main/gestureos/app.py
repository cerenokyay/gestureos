import json
import cv2

from gestureos.tracking.hands_mediapipe import HandTracker
from gestureos.gestures.rules import classify_rules
from gestureos.utils.cooldown import Cooldown
from gestureos.actions.keyboard import KeyPress, Hotkey

def build_action(spec: dict):
    t = spec.get("type")
    if t == "key_press":
        return KeyPress(spec["key"])
    if t == "hotkey":
        return Hotkey(spec["keys"])
    raise ValueError(f"Unknown action type: {t}")

def run(config_path: str = "config/default.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cam_index = int(cfg.get("camera_index", 0))
    cd = Cooldown(float(cfg.get("cooldown_sec", 0.7)))

    actions_cfg = cfg.get("actions", {})
    action_map = {g: build_action(spec) for g, spec in actions_cfg.items()}

    cap = cv2.VideoCapture(cam_index)
    tracker = HandTracker(max_hands=1)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            lm, annotated = tracker.process(frame)

            gesture = "NO_HAND"
            if lm is not None:
                gesture = classify_rules(lm)
                if gesture in action_map and cd.allow(gesture):
                    action_map[gesture].run()

            cv2.putText(annotated, f"Gesture: {gesture}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("GestureOS", annotated)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()