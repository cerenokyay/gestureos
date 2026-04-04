# GestureOS ML Enhancement Guide

Bu kılavuz, mevcut GestureOS sisteminizi ML tabanlı hareket tanıma ile geliştirmek için adım adım talimatlar içerir.

## 🎯 Genel Bakış

Mevcut sistem: **Rule-based** (kural tabanlı) hareket tanıma
Hedef sistem: **ML-based** (makine öğrenmesi) hareket tanıma

### Neden ML?
- Daha doğru tanıma
- Yeni hareketleri otomatik öğrenme
- Kişiselleştirme (kendi verilerinizle eğitim)

## 📋 Adımlar

### Step 1: Dataset Hazırlama
```bash
# Dataset klasör yapısı oluşturun
data/
└── gesture_dataset/
    ├── open_palm/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── fist/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── ...
```

### Step 2: Landmark Çıkarma
```bash
cd gesture_os/recognition
python dataset_processor.py
```
Bu script:
- Dataset görüntülerini alır
- MediaPipe ile landmark çıkarır
- `data/processed_landmarks.csv` kaydeder

### Step 3: Model Eğitimi
```bash
python ml_model.py
```
Bu script:
- RandomForest modeli eğitir
- `models/gesture_model.pkl` kaydeder
- Accuracy raporlar

### Step 4: Kendi Verilerinizi Toplayın
```bash
python data_collector.py
```
Bu script:
- Webcam'den veri toplar
- Her hareket için 100 örnek
- `data/custom_data/` klasörüne kaydeder

### Step 5: Karşılaştırma
```bash
python model_evaluator.py
```
Bu script:
- Orijinal dataset vs Custom dataset karşılaştırması
- Accuracy iyileşmesi gösterir

## 🚀 Tam Pipeline Çalıştırma

```bash
# Tek komutla tüm adımları çalıştır
python run_pipeline.py --dataset-dir /path/to/your/dataset --collect-data
```

## 📊 Beklenen Sonuçlar

| Model | Accuracy |
|-------|----------|
| Rule-based (mevcut) | ~85% |
| ML (sadece dataset) | ~90% |
| ML (dataset + kendi veri) | ~95% |

## 🔧 Entegrasyon

ML modelini ana sisteme entegre etmek için:

1. `gesture_os/main.py`'de import ekleyin:
```python
from .recognition.ml_model import GestureMLModel
```

2. `classify_rules()` yerine ML kullanın:
```python
ml_model = GestureMLModel()
gesture, confidence = ml_model.predict(lm)
```

3. Config'e ML ayarı ekleyin:
```json
{
  "use_ml": true,
  "ml_model_path": "models/gesture_model.pkl"
}
```

## 📈 İyileştirme Önerileri

1. **Daha fazla veri**: Her hareket için 500+ örnek
2. **Data augmentation**: Döndürme, ölçekleme
3. **Neural Network**: RandomForest yerine CNN
4. **Real-time fine-tuning**: Sistem kullanılırken öğrenme

## 🐛 Sorun Giderme

- **Kamera açılmıyor**: Kamera index'ini kontrol edin (`cv2.VideoCapture(0)`)
- **MediaPipe hata**: `pip install mediapipe` tekrar çalıştırın
- **Dataset bulunamadı**: Yolları kontrol edin
- **Memory error**: Batch processing kullanın

## 📞 Destek

Herhangi bir sorun yaşarsanız:
1. Hata mesajını paylaşın
2. Kullandığınız dataset yapısını belirtin
3. Sistem özelliklerinizi (OS, Python versiyonu) söyleyin