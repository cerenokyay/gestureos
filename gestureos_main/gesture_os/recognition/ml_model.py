import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from pathlib import Path

class GestureMLModel:
    def __init__(self, model_path="models/gesture_model.pkl"):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(exist_ok=True)
        self.model = None
        self.feature_columns = None

    def load_data(self, csv_path):
        """Landmark dataset'ini yükle"""
        df = pd.read_csv(csv_path)

        # Feature kolonlarını belirle (lm_0_x, lm_0_y, lm_0_z, ...)
        self.feature_columns = [col for col in df.columns if col.startswith('lm_')]

        X = df[self.feature_columns].values
        y = df['gesture'].values

        print(f"📊 Dataset yüklendi: {len(df)} örnek, {len(self.feature_columns)} özellik")
        print(f"🎯 Sınıflar: {np.unique(y)}")

        return X, y

    def train(self, csv_path, test_size=0.2, random_state=42):
        """Modeli eğit"""
        X, y = self.load_data(csv_path)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # RandomForest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )

        print("🚀 Model eğitiliyor...")
        self.model.fit(X_train, y_train)

        # Değerlendirme
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        print(f"Eğitim Accuracy: {train_score:.2f}")
        print(f"Test Accuracy: {test_score:.2f}")

        # Detaylı rapor
        y_pred = self.model.predict(X_test)
        print("\n📋 Sınıflandırma Raporu:")
        print(classification_report(y_test, y_pred))

        # Modeli kaydet
        joblib.dump(self.model, self.model_path)
        print(f"💾 Model kaydedildi: {self.model_path}")

        return train_score, test_score

    def predict(self, landmarks):
        """Tek bir landmark seti için tahmin"""
        if self.model is None:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
            else:
                raise ValueError("Model bulunamadı! Önce train() çalıştırın.")

        # Landmarks'ı düzleştir (21 nokta * 3 koordinat = 63 özellik)
        if isinstance(landmarks, np.ndarray) and landmarks.shape == (21, 3):
            features = landmarks.flatten()
        elif isinstance(landmarks, np.ndarray) and len(landmarks) == 63:
            features = landmarks
        else:
            raise ValueError(f"Geçersiz landmark format: {landmarks.shape}")

        # Tahmin
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]

        return prediction, probabilities

    def add_custom_data(self, landmarks_list, labels_list, csv_path):
        """Kendi verilerini dataset'e ekle"""
        if not os.path.exists(csv_path):
            print(f"❌ Dataset bulunamadı: {csv_path}")
            return

        df = pd.read_csv(csv_path)

        new_data = []
        for landmarks, label in zip(landmarks_list, labels_list):
            if isinstance(landmarks, np.ndarray) and landmarks.shape == (21, 3):
                flat_lm = landmarks.flatten()
            else:
                continue

            row = {'gesture': label, 'confidence': 1.0, 'image_path': 'custom_data'}
            for i, coord in enumerate(flat_lm):
                row[f'lm_{i//3}_{"xyz"[i%3]}'] = coord
            new_data.append(row)

        new_df = pd.DataFrame(new_data)
        combined_df = pd.concat([df, new_df], ignore_index=True)

        combined_df.to_csv(csv_path, index=False)
        print(f"✅ {len(new_data)} yeni örnek eklendi. Toplam: {len(combined_df)}")

        return combined_df

    def compare_models(self, original_csv, custom_csv=None):
        """Orijinal vs custom data ile eğitilmiş modelleri karşılaştır"""
        print("🔍 Model karşılaştırması...")

        # Orijinal model
        print("\n📈 Orijinal Dataset ile:")
        orig_train, orig_test = self.train(original_csv)

        if custom_csv and os.path.exists(custom_csv):
            print("\n📈 Custom Data eklendikten sonra:")
            custom_train, custom_test = self.train(custom_csv)

            improvement = custom_test - orig_test
            print(f"Model iyileşmesi: {improvement:.2f}")
        else:
            print("⚠️ Custom dataset bulunamadı")

if __name__ == "__main__":
    # Örnek kullanım
    model = GestureMLModel()

    # Dataset yolları
    processed_csv = "data/processed_landmarks.csv"
    custom_csv = "data/custom_landmarks.csv"

    if os.path.exists(processed_csv):
        # Eğitim
        model.train(processed_csv)

        # Karşılaştırma
        model.compare_models(processed_csv, custom_csv)
    else:
        print(f"❌ İşlenmiş dataset bulunamadı: {processed_csv}")
        print("💡 Önce dataset_processor.py çalıştırın!")
