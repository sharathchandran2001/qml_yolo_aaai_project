"""
main.py - Pipeline orchestration for Classical ML vs. Quantum ML & YOLOv8 framework.
"""

import sys
import numpy as np
from pathlib import Path

from config import DATA_DIR, RAW_IMG_DIR
from annotation_tool.capture import capture_screen_region
from models.classical_ml import ClassicalClassifier
from models.quantum_ml import QuantumClassifier
from models.yolo_model import YoloDetector


def run_screen_capture():
    print("\n--- Step 1: Screen Capture ---")
    save_path, img = capture_screen_region()
    print(f"Captured screen saved to: {save_path}")


def run_yolo_pipeline(epochs=5):
    print("\n--- Step 2: YOLOv8 Small Training & Inference ---")
    detector = YoloDetector(model_version="yolov8s.pt")

    # Check if dataset has training images before launching training
    train_imgs = list((DATA_DIR / "images" / "train").glob("*.png")) + \
                 list((DATA_DIR / "images" / "train").glob("*.jpg"))

    if not train_imgs:
        print("[!] No training images found in data/images/train/.")
        print("[!] Populate images and annotations before training YOLO.")
        return

    print(f"Found {len(train_imgs)} training images. Starting YOLOv8 training...")
    detector.train_model(epochs=epochs, imgsz=640)


def run_ml_comparison_demo():
    print("\n--- Step 3: Classical ML vs. Quantum ML Baseline Demo ---")

    # Dummy feature extraction for demonstration (4 features matching 4 qubits)
    np.random.seed(42)
    X_train = np.random.uniform(-np.pi, np.pi, (20, 4))
    y_train = np.random.choice([0, 1], size=(20,))

    X_test = np.random.uniform(-np.pi, np.pi, (5, 4))
    y_test = np.random.choice([0, 1], size=(5,))

    # 1. Classical SVM Classifier
    print("\n[+] Training Classical Support Vector Machine...")
    svm = ClassicalClassifier()
    svm.train(X_train, y_train)
    acc, report = svm.evaluate(X_test, y_test)
    print(f"Classical SVM Accuracy: {acc * 100:.2f}%")

    # 2. Quantum Machine Learning Classifier Initializer
    print("\n[+] Initializing Quantum Variational Circuit (PennyLane)...")
    qml_model = QuantumClassifier(n_qubits=4)
    weights = qml_model.initialize_weights(n_layers=3)
    circuit = qml_model._build_circuit()

    # Evaluate circuit output on first test sample
    q_out = circuit(weights, X_test[0])
    print(f"Sample Quantum Expectation Values (PauliZ): {q_out}")


def main():
    print("==================================================")
    print(" AAAI Research Project Core Pipeline Execution ")
    print("==================================================")

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "capture":
            run_screen_capture()
        elif mode == "yolo":
            run_yolo_pipeline()
        elif mode == "ml":
            run_ml_comparison_demo()
        else:
            print(f"Unknown mode '{mode}'. Available modes: capture, yolo, ml")
    else:
        # Run standard end-to-end check
        print("Running full pipeline test check...")
        run_ml_comparison_demo()


if __name__ == "__main__":
    main()