import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm


class LandmarkExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.3,  # 🔥 düşürdük
            min_tracking_confidence=0.3,
        )
        print("✅ MediaPipe hazır")

    def preprocess(self, image):
        h, w = image.shape[:2]
        target = 320
        if h < target or w < target:
            scale = max(target / h, target / w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Kontrasti artır, parlaklık ayarla
        image = cv2.convertScaleAbs(image, alpha=1.5, beta=20)

        # CLAHE ile daha iyi kenar/aydınlık
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Hafif gürültü azaltma
        image = cv2.bilateralFilter(image, 7, 75, 75)

        return image

    def extract_from_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None, None

        image = self.preprocess(image)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        # Eğer el bulunamazsa bir ek deneme daha yap
        if not results.multi_hand_landmarks:
            h, w = image.shape[:2]
            crop = image[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
            if crop.size > 0:
                crop = cv2.resize(crop, (max(320, crop.shape[1]), max(320, crop.shape[0])), interpolation=cv2.INTER_CUBIC)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                if results.multi_hand_landmarks:
                    image = crop

        if not results.multi_hand_landmarks:
            # Son çare: görüntüyü daha da açıp tekrar dene
            bright = cv2.convertScaleAbs(image, alpha=1.3, beta=40)
            rgb = cv2.cvtColor(bright, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None, None

        hand = results.multi_hand_landmarks[0]
        landmarks = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)

        # Normalize: ilk nokta (wrist) baz alınsın, sonra boyut eşitlemesi
        base = landmarks[0, :2].copy()
        landmarks[:, :2] -= base
        scale = np.linalg.norm(landmarks[:, :2], axis=1).max()
        if scale > 0:
            landmarks[:, :2] /= scale

        landmarks = landmarks.flatten()

        confidence = 0.0
        if results.multi_handedness:
            confidence = float(results.multi_handedness[0].classification[0].score)

        return landmarks, confidence

    def process_dataset(self, dataset_dir, output_file="processed_landmarks.csv"):
        data = []
        total_processed = 0
        total_skipped = 0

        for gesture_dir in Path(dataset_dir).iterdir():
            if not gesture_dir.is_dir():
                continue

            gesture_name = gesture_dir.name
            print(f"\n📁 İşleniyor: {gesture_name}")

            image_files = list(gesture_dir.glob("*.jpg")) + list(gesture_dir.glob("*.jpeg")) + list(gesture_dir.glob("*.png"))
            print(f"📸 {len(image_files)} resim bulundu")

            skipped_preview = 0

            for img_path in tqdm(image_files, desc=gesture_name):
                landmarks, confidence = self.extract_from_image(str(img_path))

                if landmarks is not None:
                    row = {
                        "gesture": gesture_name,
                        "confidence": confidence
                    }

                    # 63 feature (21 * xyz)
                    for i, val in enumerate(landmarks):
                        row[f"f_{i}"] = val

                    data.append(row)
                    total_processed += 1
                else:
                    total_skipped += 1
                    if skipped_preview < 5:
                        print(f"⚠️ Atlandı: {img_path.name}")
                        skipped_preview += 1

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

        print("\n✅ BİTTİ")
        print(f"📊 İşlenen: {total_processed}")
        print(f"❌ Atlanan: {total_skipped}")
        print(f"💾 Kayıt: {output_file}")
        print(f"🎯 Sınıflar: {df['gesture'].unique()}")

        return df

    def close(self):
        self.hands.close()


if __name__ == "__main__":
    extractor = LandmarkExtractor()

    dataset_path = "data/images"  # 🔥 kendi veri setin için burası doğru yol

    if os.path.exists(dataset_path):
        extractor.process_dataset(dataset_path, "data/processed_landmarks.csv")
    else:
        print(f"❌ Dataset bulunamadı: {dataset_path}")
        print("💡 Yeni veri setin için şu yapıyı kullan:")
        print("data/images/")
        print("├── closedFist/")
        print("├── fingerCircle/")
        print("├── ...")

    extractor.close()