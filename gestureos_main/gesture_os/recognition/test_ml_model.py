#!/usr/bin/env python3
"""
ML Model Test Script
Eğitilmiş modeli gerçek zamanlı test etmek için
"""

import cv2
import numpy as np
import time

from ..vision.hand_tracker import HandTracker
from ..recognition.ml_model import GestureMLModel
from ..recognition.rule_based import classify_rules

class MLTester:
    def __init__(self, model_path="models/gesture_model.pkl"):
        self.tracker = HandTracker(max_hands=1)
        self.ml_model = GestureMLModel(model_path)

        # Test istatistikleri
        self.stats = {
            'total_frames': 0,
            'ml_predictions': 0,
            'rule_predictions': 0,
            'matches': 0,
            'ml_correct': 0,
            'rule_correct': 0
        }

    def test_realtime(self, camera_index=0, duration_sec=30):
        """Gerçek zamanlı test"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Kamera {camera_index} açılamadı!")
            return

        print("🎬 ML Model Testi başlatılıyor...")
        print(f"⏱️ Süre: {duration_sec} saniye")
        print("💡 Hareket yapın ve sonuçları karşılaştırın")
        print("❌ Çıkmak için 'Q' basın")

        start_time = time.time()
        last_gesture = None

        while time.time() - start_time < duration_sec:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            lm, annotated, mouse_pos, confidence = self.tracker.process(frame)

            self.stats['total_frames'] += 1

            ml_gesture = "NO_HAND"
            rule_gesture = "NO_HAND"

            if lm is not None and confidence > 0.5:
                # ML tahmin
                try:
                    ml_gesture, probabilities = self.ml_model.predict(lm)
                    self.stats['ml_predictions'] += 1
                except Exception as e:
                    ml_gesture = f"ML_ERROR: {str(e)[:20]}"

                # Rule-based tahmin
                rule_gesture = classify_rules(lm)
                self.stats['rule_predictions'] += 1

                # Karşılaştırma
                if ml_gesture == rule_gesture and ml_gesture != "NO_HAND":
                    self.stats['matches'] += 1

                # Ground truth (kullanıcı input) - basit versiyon
                if last_gesture != ml_gesture and ml_gesture not in ["NO_HAND", "UNKNOWN"]:
                    print(f"🎯 ML: {ml_gesture} | Rule: {rule_gesture} | Conf: {confidence:.2f}")
                    last_gesture = ml_gesture

            # UI
            cv2.putText(annotated, f"ML: {ml_gesture}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(annotated, f"Rule: {rule_gesture}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            cv2.putText(annotated, f"Conf: {confidence:.2f}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            remaining = int(duration_sec - (time.time() - start_time))
            cv2.putText(annotated, f"Kalan: {remaining}s", (20, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("ML Model Test", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        self.print_stats()

    def print_stats(self):
        """Test istatistiklerini yazdır"""
        print("\n📊 TEST İSTATİSTİKLERİ")
        print("="*40)
        print(f"Toplam frame: {self.stats['total_frames']}")
        print(f"ML tahmin sayısı: {self.stats['ml_predictions']}")
        print(f"Rule tahmin sayısı: {self.stats['rule_predictions']}")
        print(f"Eşleşen tahminler: {self.stats['matches']}")

        if self.stats['ml_predictions'] > 0:
            match_rate = self.stats['matches'] / self.stats['ml_predictions'] * 100
            print(".1f")
        print("\n💡 Değerlendirme:")
        if self.stats['matches'] > 0:
            print("✅ ML ve Rule-based sistemler uyumlu çalışıyor")
        else:
            print("⚠️ ML modeli rule-based'den farklı sonuçlar üretiyor")

    def compare_accuracy(self, test_data_csv):
        """CSV test verisi ile accuracy karşılaştırma"""
        import pandas as pd

        if not pd.io.common.file_exists(test_data_csv):
            print(f"❌ Test dosyası bulunamadı: {test_data_csv}")
            return

        df = pd.read_csv(test_data_csv)
        feature_cols = [col for col in df.columns if col.startswith('lm_')]

        correct_ml = 0
        correct_rule = 0
        total = 0

        print("🔍 Test verisi ile accuracy karşılaştırması...")

        for _, row in df.iterrows():
            landmarks = row[feature_cols].values
            true_gesture = row['gesture']

            # ML tahmin
            try:
                ml_pred, _ = self.ml_model.predict(landmarks.reshape(21, 3))
                if ml_pred == true_gesture:
                    correct_ml += 1
            except:
                pass

            # Rule-based tahmin (landmark'dan gesture'a çevirmek için basit yaklaşım)
            # Bu kısım gerçek rule-based logic'e göre uyarlanmalı
            rule_pred = "UNKNOWN"  # Placeholder
            if rule_pred == true_gesture:
                correct_rule += 1

            total += 1

        print(f"📊 Test sonuçları ({total} örnek):")
        print(".1f")
        print(".1f")
    def close(self):
        self.tracker.close()

if __name__ == "__main__":
    # Örnek kullanım
    tester = MLTester()

    print("🎯 ML Model Test Modu")
    print("1. Gerçek zamanlı test (30 saniye)")
    print("2. CSV test verisi ile karşılaştırma")

    choice = input("Seçiminiz (1/2): ").strip()

    if choice == "1":
        tester.test_realtime(duration_sec=30)
    elif choice == "2":
        csv_path = input("Test CSV dosya yolu: ").strip()
        if csv_path:
            tester.compare_accuracy(csv_path)
    else:
        print("❌ Geçersiz seçim")

    tester.close()