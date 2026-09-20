import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from config import NUM_QUBITS, NUM_LAYERS, LEARNING_RATE, EPOCHS, PROCESSED_DATA_DIR, DATA_DIR, CLASS_NAMES
from models.yolo_model import YOLOCropExtractor
from models.quantum_ml import QuantumClassifierModel

# ==============================================================================
# 1. SHARED PREPROCESSING & FIXED PCA SCALER
# ==============================================================================
class SharedPCATransformer:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scale_max = 1.0

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        X_pca = self.pca.fit_transform(X_train)
        self.scale_max = np.max(np.abs(X_pca))
        if self.scale_max == 0:
            self.scale_max = 1.0
        return np.pi * (X_pca / self.scale_max)

    def transform(self, X_test: np.ndarray) -> np.ndarray:
        X_pca = self.pca.transform(X_test)
        return np.pi * (X_pca / self.scale_max)

# ==============================================================================
# 2. CLASS BALANCING HELPER
# ==============================================================================
def balance_dataset(X: np.ndarray, y: np.ndarray):
    """
    Oversamples the minority class (calculator_display) so classes are 1:1 balanced.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y

    max_count = np.max(counts)
    balanced_X, balanced_y = [], []

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        resampled_idx = np.random.choice(cls_idx, size=max_count, replace=True)
        balanced_X.append(X[resampled_idx])
        balanced_y.append(y[resampled_idx])

    X_bal = np.vstack(balanced_X)
    y_bal = np.hstack(balanced_y)
    
    # Shuffle dataset
    shuffle_idx = np.random.permutation(len(y_bal))
    return X_bal[shuffle_idx], y_bal[shuffle_idx]

# ==============================================================================
# 3. ROBUST LATENCY BENCHMARKING
# ==============================================================================
def measure_robust_latency(predict_fn, sample: np.ndarray, n_warmup: int = 50, n_runs: int = 200):
    for _ in range(n_warmup):
        _ = predict_fn(sample)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = predict_fn(sample)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return np.mean(latencies), np.std(latencies)

# ==============================================================================
# 4. PLOTTING HELPERS
# ==============================================================================
def save_confusion_matrices(confusion_matrices, save_path):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle("Confusion Matrices Across Models", fontsize=14, fontweight='bold')

    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_benchmark_chart(results, save_path):
    models = list(results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in models]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_ylabel('Score')
    ax.set_title('Classical vs. Quantum Model Comparison (Strict Image-Level Split)', fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==============================================================================
# 5. BENCHMARK EXECUTION PIPELINE
# ==============================================================================
def run_rigorous_benchmark():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print("[+] Initializing Strict Image-Level Benchmark Pipeline...")

    extractor = YOLOCropExtractor()

    # Strict Image-Level Split
    train_img_dir = os.path.join(DATA_DIR, "images", "train")
    val_img_dir = os.path.join(DATA_DIR, "images", "val")

    X_train_raw, y_train_raw = extractor.extract_all_crop_features(images_dir=train_img_dir)
    X_test_1024, y_test = extractor.extract_all_crop_features(images_dir=val_img_dir)

    if len(X_train_raw) == 0 or len(X_test_1024) == 0:
        raise ValueError("[-] Feature extraction failed. Ensure images exist in data/images/train and data/images/val.")

    # Balance Training Set
    X_train_1024, y_train = balance_dataset(X_train_raw, y_train_raw)

    print(f"[+] Dataset Split (Strict Image-Level): {len(X_train_1024)} Balanced Train Crops | {len(X_test_1024)} Test Crops")

    # Shared PCA Compression Stage
    pca_scaler = SharedPCATransformer(n_components=NUM_QUBITS)
    X_train_4d = pca_scaler.fit_transform(X_train_1024)
    X_test_4d = pca_scaler.transform(X_test_1024)

    models = {
        "SVM (1024-D)": SVC(kernel='rbf', C=1.0),
        "Random Forest (1024-D)": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (4-D PCA)": SVC(kernel='rbf', C=1.0),
        "Random Forest (4-D PCA)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Quantum VQC (4-D PCA)": QuantumClassifierModel(
            n_qubits=NUM_QUBITS, n_layers=NUM_LAYERS, lr=LEARNING_RATE, epochs=EPOCHS
        )
    }

    results = {}
    confusion_matrices = {}

    for name, model in models.items():
        print(f"\n[+] Training Model: {name}...")
        
        if "1024-D" in name:
            X_tr, X_te = X_train_1024, X_test_1024
        else:
            X_tr, X_te = X_train_4d, X_test_4d

        if "Quantum" in name:
            model.fit_preprocessed(X_tr, y_train)
            preds = model.predict_preprocessed(X_te)
            predict_fn = lambda x: model.predict_preprocessed(x)
        else:
            model.fit(X_tr, y_train)
            preds = model.predict(X_te)
            predict_fn = lambda x: model.predict(x)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro', zero_division=0)
        rec = recall_score(y_test, preds, average='macro', zero_division=0)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, preds)

        sample_input = X_te[0:1]
        mean_lat, std_lat = measure_robust_latency(predict_fn, sample_input)

        results[name] = {
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1,
            "Latency (ms)": mean_lat, "Latency Std": std_lat
        }
        confusion_matrices[name] = cm

    print("\n" + "=" * 88)
    print(f"{'RIGOROUS AAAI RESEARCH COMPARATIVE BENCHMARK':^88}")
    print("=" * 88)
    print(f"{'Model Architecture':<25} | {'Acc':<7} | {'Prec':<7} | {'Rec':<7} | {'F1-Score':<8} | {'Latency (ms)':<16}")
    print("-" * 88)
    for name, m in results.items():
        lat_str = f"{m['Latency (ms)']:.3f} ± {m['Latency Std']:.3f}"
        print(f"{name:<25} | {m['Accuracy']:<7.4f} | {m['Precision']:<7.4f} | {m['Recall']:<7.4f} | {m['F1-Score']:<8.4f} | {lat_str:<16}")
    print("=" * 88)

    # Save Plots
    cm_path = os.path.join(PROCESSED_DATA_DIR, "confusion_matrices.png")
    chart_path = os.path.join(PROCESSED_DATA_DIR, "classical_vs_qml_benchmark.png")

    save_confusion_matrices(confusion_matrices, cm_path)
    save_benchmark_chart(results, chart_path)

    print(f"\n[+] Confusion matrices saved to: {cm_path}")
    print(f"[+] Benchmark metrics chart saved to: {chart_path}")

if __name__ == "__main__":
    run_rigorous_benchmark()