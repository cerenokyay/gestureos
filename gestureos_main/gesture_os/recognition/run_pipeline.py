#!/usr/bin/env python3
"""
GestureOS ML Pipeline Runner
Dataset işleme, model eğitimi ve değerlendirme için ana script
"""

import argparse
import os
from pathlib import Path
import sys

# Modülleri import et
from .dataset_processor import LandmarkExtractor
from .ml_model import GestureMLModel
from .data_collector import DataCollector
from .model_evaluator import ModelEvaluator

def run_dataset_processing(dataset_dir, output_file="data/processed_landmarks.csv"):
    """Step 1: Dataset'i işle ve landmark çıkar"""
    print("🔄 STEP 1: Dataset İşleme")
    print("="*50)

    extractor = LandmarkExtractor()
    df = extractor.process_dataset(dataset_dir, output_file)
    extractor.close()

    return output_file if df is not None else None

def run_model_training(csv_path, model_path="models/gesture_model.pkl"):
    """Step 2: Model eğitimi"""
    print("\n🔄 STEP 2: Model Eğitimi")
    print("="*50)

    model = GestureMLModel(model_path)
    train_acc, test_acc = model.train(csv_path)

    return model

def run_data_collection(gestures_list=None, output_dir="data/custom_data"):
    """Step 3: Kendi verilerini topla"""
    print("\n🔄 STEP 3: Veri Toplama")
    print("="*50)

    if gestures_list is None:
        gestures_list = ["OPEN_PALM", "FIST", "POINT", "THUMBS_UP", "PINCH"]

    collector = DataCollector(output_dir)
    data = collector.collect_multiple_gestures(gestures_list)

    if data:
        csv_path = collector.save_to_csv()
        json_path = collector.save_to_json()
        collector.close()
        return csv_path
    else:
        collector.close()
        return None

def run_model_comparison(original_csv, custom_csv=None):
    """Step 4: Model karşılaştırması"""
    print("\n🔄 STEP 4: Model Karşılaştırması")
    print("="*50)

    evaluator = ModelEvaluator()
    results = evaluator.compare_datasets(original_csv, custom_csv)

    return results

def main():
    parser = argparse.ArgumentParser(description="GestureOS ML Pipeline")
    parser.add_argument("--dataset-dir", help="Ham dataset klasörü yolu")
    parser.add_argument("--processed-csv", default="data/processed_landmarks.csv",
                       help="İşlenmiş landmark CSV dosyası")
    parser.add_argument("--custom-csv", default="data/custom_data/custom_landmarks.csv",
                       help="Custom veri CSV dosyası")
    parser.add_argument("--model-path", default="models/gesture_model.pkl",
                       help="Model dosyası yolu")
    parser.add_argument("--collect-data", action="store_true",
                       help="Webcam'den veri topla")
    parser.add_argument("--gestures", nargs="+",
                       default=["OPEN_PALM", "FIST", "POINT", "THUMBS_UP", "PINCH"],
                       help="Toplanacak hareketler")

    args = parser.parse_args()

    # Data klasörlerini oluştur
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    processed_csv = args.processed_csv
    custom_csv = args.custom_csv

    # Step 1: Dataset işleme
    if args.dataset_dir and os.path.exists(args.dataset_dir):
        processed_csv = run_dataset_processing(args.dataset_dir, processed_csv)
        if not processed_csv:
            print("❌ Dataset işleme başarısız!")
            sys.exit(1)

    # Step 3: Veri toplama (istendiği takdirde)
    if args.collect_data:
        custom_csv = run_data_collection(args.gestures)
        if not custom_csv:
            print("⚠️ Veri toplama atlandı veya başarısız")

    # Step 2: Model eğitimi
    if os.path.exists(processed_csv):
        model = run_model_training(processed_csv, args.model_path)
    else:
        print(f"❌ İşlenmiş dataset bulunamadı: {processed_csv}")
        sys.exit(1)

    # Step 4: Karşılaştırma
    if os.path.exists(processed_csv):
        results = run_model_comparison(processed_csv, custom_csv if os.path.exists(custom_csv) else None)
    else:
        print("❌ Karşılaştırma için dataset bulunamadı!")

    print("\n🎉 Pipeline tamamlandı!")
    print("💡 Sonraki adımlar:")
    print("   1. Model dosyasını gesture_os/recognition/ altına kopyala")
    print("   2. main.py'de ml_model.py'yi import et ve kullan")
    print("   3. Test et ve accuracy'yi karşılaştır")

if __name__ == "__main__":
    main()