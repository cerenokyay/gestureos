import json
import argparse
import cv2
from pathlib import Path

from .vision.hand_tracker import HandTracker
from .recognition.rule_based import classify_rules
from .actions.executor import Cooldown, KeyPress, Hotkey, Message


def build_action(spec: dict):
    t = spec.get("type")
    if t == "key_press":
        return KeyPress(spec["key"])
    if t == "hotkey":
        return Hotkey(spec["keys"])
    if t == "message":
        return Message(spec["text"])
    raise ValueError(f"Unknown action type: {t}")


def run(config_path: str = None):
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "mappings.json"
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        print(f"❌ Hata: Config dosyası bulunamadı: {config_path}")
        return
    except json.JSONDecodeError:
        print(f"❌ Hata: Config dosyası geçersiz JSON: {config_path}")
        return

    cam_index = int(cfg.get("camera_index", 0))
    cd = Cooldown(float(cfg.get("cooldown_sec", 0.7)))

    actions_cfg = cfg.get("actions", {})
    action_map = {g: build_action(spec) for g, spec in actions_cfg.items()}

    cap = cv2.VideoCapture(cam_index)
    
    if not cap.isOpened():
        print(f"❌ Hata: Kamera {cam_index} açılamadı!")
        print("💡 İpucu: Kameranın bağlı olduğundan ve başka bir uygulamada kullanılmadığından emin ol")
        return
    
    print(f"✓ Kamera {cam_index} başarıyla açıldı")
    print(f"✓ Eylemler yüklendi: {', '.join(action_map.keys())}")
    print("⏳ Çalışıyor... (ESC tuşuna basarak çık)")
    
    tracker = HandTracker(max_hands=1)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("❌ Hata: Kameradan frame alınamadı")
                break

            frame = cv2.flip(frame, 1)
            lm, annotated = tracker.process(frame)

            gesture = "NO_HAND"
            if lm is not None:
                gesture = classify_rules(lm)
                if gesture in action_map and cd.allow(gesture):
                    print(f"🎯 Hareket tespit edildi: {gesture}")
                    action_map[gesture].run()

            cv2.putText(annotated, f"Gesture: {gesture}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("GestureOS", annotated)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("✓ Çıkılıyor...")
                break
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Kapatıldı")


def main():
    p = argparse.ArgumentParser(prog="gesture_os")
    p.add_argument("--config", default=None, help="Path to config JSON (default: gesture_os/config/mappings.json)")
    args = p.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
