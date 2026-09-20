import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from config import NUM_QUBITS, NUM_LAYERS, LEARNING_RATE, EPOCHS, PROCESSED_DATA_DIR
from models.yolo_model import YOLOCropExtractor
from models.quantum_ml import QuantumClassifierModel

# ==============================================================================
# 1. SHARED PREPROCESSING & FIXED PCA SCALER
# ==============================================================================
class SharedPCATransformer:
    """
    Shared PCA Transformer that isolates feature reduction from classifier models
    and enforces strict training-bound scaling to prevent test-batch leakage.
    """
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scale_max = 1.0

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fits PCA on X_train and calculates scaling bound."""
        X_pca = self.pca.fit_transform(X_train)
        self.scale_max = np.max(np.abs(X_pca))
        if self.scale_max == 0:
            self.scale_max = 1.0
        # Map to angular domain [-pi, pi] for quantum gate encoding
        X_scaled = np.pi * (X_pca / self.scale_max)
        return X_scaled

    def transform(self, X_test: np.ndarray) -> np.ndarray:
        """Transforms X_test using pre-computed training parameters and scale_max."""
        X_pca = self.pca.transform(X_test)
        # Fixes Scaling Leakage: Uses training scale_max rather than test batch max
        X_scaled = np.pi * (X_pca / self.scale_max)
        return X_scaled


# ==============================================================================
# 2. ROBUST LATENCY BENCHMARKING FUNCTION
# ==============================================================================
def measure_robust_latency(predict_fn, sample: np.ndarray, n_warmup: int = 50, n_runs: int = 200):
    """
    Measures inference latency using high-resolution time.perf_counter() with
    warmup iterations to eliminate OS thread scheduling and cache noise.
    """
    # Warmup phase
    for _ in range(n_warmup):
        _ = predict_fn(sample)

    # Measurement phase
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = predict_fn(sample)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # Convert to milliseconds

    mean_ms = np.mean(latencies)
    std_ms = np.std(latencies)
    return mean_ms, std_ms


# ==============================================================================
# 3. BENCHMARK EXECUTION PIPELINE
# ==============================================================================
def run_rigorous_benchmark():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print("[+] Initializing Rigorous Classical vs. Quantum Benchmark Pipeline...")

    # Step A: Feature Extraction
    extractor = YOLOCropExtractor()
    features, labels = extractor.extract_all_crop_features()

    if len(features) == 0:
        raise ValueError("[-] No crop features extracted. Ensure images/labels exist in the dataset directory.")

    # Step B: Train/Test Split
    X_train_1024, X_test_1024, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    print(f"[+] Dataset Split: {len(X_train_1024)} Train Samples | {len(X_test_1024)} Test Samples")

    # Step C: Shared PCA Compression (Isolated Stage)
    print(f"[+] Fitting Shared PCA ({X_train_1024.shape[1]}-D -> {NUM_QUBITS}-D)...")
    pca_scaler = SharedPCATransformer(n_components=NUM_QUBITS)
    X_train_4d = pca_scaler.fit_transform(X_train_1024)
    X_test_4d = pca_scaler.transform(X_test_1024)

    # Step D: Initialize Models
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

    # Step E: Train and Evaluate Models
    for name, model in models.items():
        print(f"\n[+] Training Model: {name}...")
        
        # Select appropriate input feature space
        if "1024-D" in name:
            X_tr, X_te = X_train_1024, X_test_1024
        else:
            X_tr, X_te = X_train_4d, X_test_4d

        # Model Training & Inference
        if "Quantum" in name:
            # Train VQC directly on pre-transformed 4-D features
            model.fit_preprocessed(X_tr, y_train)
            preds = model.predict_preprocessed(X_te)
            predict_fn = lambda x: model.predict_preprocessed(x)
        else:
            model.fit(X_tr, y_train)
            preds = model.predict(X_te)
            predict_fn = lambda x: model.predict(x)

        # Compute Metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro', zero_division=0)
        rec = recall_score(y_test, preds, average='macro', zero_division=0)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, preds)

        # Measure Robust Latency (using 1 test sample)
        sample_input = X_te[0:1]
        mean_lat, std_lat = measure_robust_latency(predict_fn, sample_input, n_warmup=50, n_runs=200)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Latency (ms)": mean_lat,
            "Latency Std": std_lat
        }
        confusion_matrices[name] = cm

    # Step F: Display Metric Summary Table
    print("\n" + "=" * 88)
    print(f"{'RIGOROUS AAAI RESEARCH COMPARATIVE BENCHMARK':^88}")
    print("=" * 88)
    print(f"{'Model Architecture':<25} | {'Acc':<7} | {'Prec':<7} | {'Rec':<7} | {'F1-Score':<8} | {'Latency (ms)':<16}")
    print("-" * 88)
    for name, m in results.items():
        lat_str = f"{m['Latency (ms)']:.3f} ± {m['Latency Std']:.3f}"
        print(f"{name:<25} | {m['Accuracy']:<7.4f} | {m['Precision']:<7.4f} | {m['Recall']:<7.4f} | {m['F1-Score']:<8.4f} | {lat_str:<16}")
    print("=" * 88)

    # Step G: Plot and Save Confusion Matrices
    plot_confusion_matrices(confusion_matrices)

    # Step H: Plot and Save Benchmark Comparison Chart
    plot_benchmark_metrics(results)


# ==============================================================================
# 4. VISUALIZATION HELPERS
# ==============================================================================
def plot_confusion_matrices(cms: dict):
    """Plots and saves confusion matrices for all evaluated models in a grid."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    model_names = list(cms.keys())

    for idx, name in enumerate(model_names):
        sns.heatmap(cms[name], annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
        axes[idx].set_title(name, fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')

    plt.tight_layout()
    cm_path = os.path.join(PROCESSED_DATA_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrices saved to: {cm_path}")


def plot_benchmark_metrics(results: dict):
    """Plots comparative metric bar chart incorporating both 1024-D and 4-D baselines."""
    labels = list(results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    x = np.arange(len(metrics))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_name in enumerate(labels):
        values = [results[model_name][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=model_name)

    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=12)
    ax.set_title('Apples-to-Apples Performance Comparison (1024-D Full vs. 4-D PCA vs. QML)', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0.0, 1.1)
    ax.legend(loc='lower right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    chart_path = os.path.join(PROCESSED_DATA_DIR, "classical_vs_qml_benchmark.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[+] Benchmark metrics chart saved to: {chart_path}")


if __name__ == "__main__":
    run_rigorous_benchmark()