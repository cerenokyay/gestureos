import cv2
import numpy as np
import time
from pathlib import Path
import json
from collections import defaultdict

from ..vision.hand_tracker import HandTracker
from ..recognition.rule_based import classify_rules

class DataCollector:
    def __init__(self, output_dir="data/custom_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.tracker = HandTracker(max_hands=1)
        self.collected_data = defaultdict(list)

        # Veri toplama ayarları
        self.samples_per_gesture = 100  # Her hareket için kaç örnek
        self.collection_delay = 0.1     # Kareler arası bekleme (saniye)

    def collect_gesture_data(self, gesture_name, camera_index=0):
        """Belirli bir hareket için veri topla"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Kamera {camera_index} açılamadı!")
            return []

        print(f"🎯 '{gesture_name}' hareketi için veri toplanıyor...")
        print(f"📊 Hedef: {self.samples_per_gesture} örnek")
        print("💡 Hareketi yapın ve sabit tutun. 'Q' ile çıkın.")

        collected = []
        frame_count = 0

        while len(collected) < self.samples_per_gesture:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            lm, annotated, mouse_pos, confidence = self.tracker.process(frame)

            current_gesture = "NO_HAND"
            if lm is not None and confidence > 0.5:
                current_gesture = classify_rules(lm)

                # Sadece istediğimiz hareketi kaydet
                if current_gesture == gesture_name:
                    collected.append({
                        'landmarks': lm.copy(),
                        'confidence': confidence,
                        'timestamp': time.time()
                    })

                    # Görsel feedback
                    cv2.putText(annotated, f"Kaydedildi: {len(collected)}/{self.samples_per_gesture}",
                              (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # UI
            cv2.putText(annotated, f"Hareket: {current_gesture}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            cv2.putText(annotated, f"Toplanan: {len(collected)}/{self.samples_per_gesture}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Data Collection", annotated)

            # Çıkış kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            frame_count += 1
            time.sleep(self.collection_delay)

        cap.release()
        cv2.destroyAllWindows()

        print(f"✅ '{gesture_name}' için {len(collected)} örnek toplandı")
        return collected

    def collect_multiple_gestures(self, gestures_list):
        """Birden fazla hareket için veri topla"""
        all_data = {}

        for gesture in gestures_list:
            print(f"\n🔄 {gesture} hareketi için hazırlanın...")
            input("Enter'a basın başlayın...")

            data = self.collect_gesture_data(gesture)
            if data:
                all_data[gesture] = data
                self.collected_data[gesture] = data

        return all_data

    def save_to_csv(self, filename="custom_landmarks.csv"):
        """Toplanan veriyi CSV olarak kaydet"""
        import pandas as pd

        all_rows = []
        for gesture, samples in self.collected_data.items():
            for sample in samples:
                lm = sample['landmarks']
                if lm.shape == (21, 3):
                    flat_lm = lm.flatten()
                    row = {
                        'gesture': gesture,
                        'confidence': sample['confidence'],
                        'image_path': 'webcam_capture'
                    }
                    # Landmark'ları ekle
                    for i, coord in enumerate(flat_lm):
                        row[f'lm_{i//3}_{"xyz"[i%3]}'] = coord
                    all_rows.append(row)

        if all_rows:
            df = pd.DataFrame(all_rows)
            output_path = self.output_dir / filename
            df.to_csv(output_path, index=False)
            print(f"💾 Veri kaydedildi: {output_path}")
            print(f"📊 Toplam örnek: {len(df)}")
            return str(output_path)
        else:
            print("❌ Kaydedilecek veri bulunamadı!")
            return None

    def save_to_json(self, filename="custom_data.json"):
        """Veriyi JSON olarak kaydet (daha detaylı)"""
        output_path = self.output_dir / filename

        # Numpy array'leri listeye çevir
        serializable_data = {}
        for gesture, samples in self.collected_data.items():
            serializable_data[gesture] = []
            for sample in samples:
                serializable_data[gesture].append({
                    'landmarks': sample['landmarks'].tolist(),
                    'confidence': sample['confidence'],
                    'timestamp': sample['timestamp']
                })

        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        print(f"💾 JSON veri kaydedildi: {output_path}")
        return str(output_path)

    def close(self):
        self.tracker.close()

if __name__ == "__main__":
    # Örnek kullanım
    collector = DataCollector()

    # Toplanacak hareketler
    gestures_to_collect = [
        "OPEN_PALM",
        "FIST",
        "POINT",
        "THUMBS_UP",
        "PINCH"
    ]

    print("🎬 Webcam veri toplama başlatılıyor...")
    print("💡 Her hareket için hazırlanın ve Enter'a basın.")

    # Veri topla
    data = collector.collect_multiple_gestures(gestures_to_collect)

    # Kaydet
    if data:
        csv_path = collector.save_to_csv()
        json_path = collector.save_to_json()

        print("\n✅ Veri toplama tamamlandı!")
        print(f"📁 CSV: {csv_path}")
        print(f"📁 JSON: {json_path}")

    collector.close()