import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from .ml_model import GestureMLModel

class ModelEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate_model(self, model, X, y, model_name="Model"):
        """Modeli değerlendir"""
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        mean_cv = np.mean(cv_scores)
        std_cv = np.std(cv_scores)

        # Train/test split ile accuracy
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        result = {
            'model_name': model_name,
            'cv_mean': mean_cv,
            'cv_std': std_cv,
            'test_accuracy': test_acc,
            'cv_scores': cv_scores
        }

        self.results[model_name] = result
        return result

    def compare_datasets(self, original_csv, custom_csv=None):
        """İki dataset'i karşılaştır"""
        print("🔍 Dataset karşılaştırması...")

        # Orijinal dataset
        orig_model = GestureMLModel()
        X_orig, y_orig = orig_model.load_data(original_csv)
        orig_result = self.evaluate_model(orig_model.model, X_orig, y_orig, "Orijinal Dataset")

        results = {'original': orig_result}

        if custom_csv and pd.io.common.file_exists(custom_csv):
            # Custom dataset
            custom_model = GestureMLModel()
            X_custom, y_custom = custom_model.load_data(custom_csv)
            custom_result = self.evaluate_model(custom_model.model, X_custom, y_custom, "Custom + Orijinal")

            results['custom'] = custom_result

            # İyileşme
            improvement = custom_result['test_accuracy'] - orig_result['test_accuracy']
            results['improvement'] = improvement

            print(".2f")
            print(".2f")
            print(".2f")
        else:
            print("⚠️ Custom dataset bulunamadı, sadece orijinal ile karşılaştırma")

        self.print_comparison_table(results)
        return results

    def print_comparison_table(self, results):
        """Karşılaştırma tablosunu yazdır"""
        print("\n📊 MODEL KARŞILAŞTIRMA TABLOSU")
        print("="*60)
        print("<15")
        print("-"*60)

        for key, result in results.items():
            if key == 'improvement':
                continue
            print("<15"
                  "<8.3f"
                  "<8.3f"
                  "<8.3f")

        if 'improvement' in results:
            print("-"*60)
            print("<15"
                  "<8.3f")

    def plot_confusion_matrix(self, model, X, y, labels, title="Confusion Matrix"):
        """Confusion matrix çiz"""
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f"models/{title.lower().replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
        plt.show()

    def detailed_report(self, model, X, y, model_name="Model"):
        """Detaylı sınıflandırma raporu"""
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n📋 {model_name} - Detaylı Sınıflandırma Raporu")
        print("="*50)
        print(classification_report(y_test, y_pred))

        # Her sınıf için accuracy
        unique_labels = np.unique(y_test)
        self.plot_confusion_matrix(model, X, y, unique_labels,
                                 f"{model_name} Confusion Matrix")

if __name__ == "__main__":
    # Örnek kullanım
    evaluator = ModelEvaluator()

    # Dataset'leri karşılaştır
    original_csv = "data/processed_landmarks.csv"
    custom_csv = "data/custom_data/custom_landmarks.csv"

    if pd.io.common.file_exists(original_csv):
        results = evaluator.compare_datasets(original_csv, custom_csv)

        # Detaylı rapor
        orig_model = GestureMLModel()
        X, y = orig_model.load_data(original_csv)
        evaluator.detailed_report(orig_model.model, X, y, "Orijinal Dataset")
    else:
        print(f"❌ Dataset bulunamadı: {original_csv}")