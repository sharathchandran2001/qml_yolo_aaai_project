qml_yolo_aaai_project/
│
├── annotation_tool/             # Dataset collection & manual labeling tools
│   ├── capture.py               # Utility: Captures active screen/calculator windows
│   └── coordinate_tagger.py     # Utility: Converts screen coordinates to YOLO label format
│
├── models/                      # Dual-path classification architectures
│   ├── classical_ml.py          # Classical Classifiers (SVM & Random Forest on 1024-D)
│   ├── quantum_ml.py            # PennyLane QML Classifier (4-Qubit VQC on 4-D PCA)
│   └── yolo_model.py            # Ultralytics YOLOv8 object detector interface
│
├── data/                        # Structured dataset directory (images & labels)[cite: 4]
├── runs/                        # YOLO training run logs and best.pt weights[cite: 4]
├── venv/                        # Python virtual environment[cite: 4]
│
├── benchmark_compare.py         # CORE: Main end-to-end benchmark & evaluation script[cite: 4]
├── classical_pipeline.py        # CORE: Single-image inspection & classical test harness[cite: 4]
├── config.py                    # CORE: Central hyperparameters & file path configurations[cite: 4]
├── generate_labels.py           # CORE: Automated YOLO label dataset generator[cite: 4]
├── main.py                      # CORE: CLI entry point for training YOLO detector[cite: 4]
│
├── pipeline_diagram.py          # HELPER: Generator script for Mermaid pipeline PNGs[cite: 4]
├── pipeline_diagram_1.py        # HELPER: Generator script for 4-stage workflow PNGs[cite: 4]
│
├── pipeline_diagram.png         # Artifact: Rendered architecture diagram[cite: 4]
├── updated_pipeline_workflow.png# Artifact: Rendered 4-stage workflow diagram[cite: 4]
├── yolov8s.pt                   # Pretrained YOLOv8 Small weights[cite: 4]
├── README.md                    # Project overview[cite: 4]
├── RESEARCH.md                  # AAAI research draft text[cite: 4]
├── requirements.txt             # Environment dependencies[cite: 4]
└── .gitignore                   # Git exclusion rules[cite: 4]

    



=======
### Classical ML
place cacl images in train under images
python .\generate_labels.py
--> will generate labels under images/val
python main.py yolo
--> download and train model best.pt
python classical_pipeline.py
-->
 Crop #1 | BBox: (85, 325, 165, 530) | YOLO Detected: calculator_display | SVM Predicted: calculator_display
 Crop #2 | BBox: (3, 327, 86, 533) | YOLO Detected: calculator_display | SVM Predicted: calculator_display



### QML & Benchmark Architecture

python benchmark_compare.py
--->
Conclusion
Dimensional Compression Efficiency ($1024 \to 4$): The Variational Quantum Classifier (VQC) achieved $97.40\%$ accuracy and an F1-Score of $0.9739$ using only 4 qubits (4 PCA features compressed from the original 1024-dimensional YOLO crop features). This proves that the low-dimensional quantum state space retains enough salient semantic information to distinguish detected components.
Loss Convergence: 
The loss dropped from $3.8354 \to 0.1277$ across 15 epochs, demonstrating stable parameter updates for the StronglyEntanglingLayers under the PennyLane Adam Optimizer.
Inference Latency Trade-off: 
Classical SVM ($0.03\text{ ms}$) and Random Forest ($0.11\text{ ms}$) run faster than Quantum VQC ($5.97\text{ ms}$) due to classical matrix multiplication efficiency versus qubit state vector simulation overhead on CPU. This performance gap highlights current NISQ simulator constraints and sets up the standard narrative for AAAI papers comparing classical vs. simulated quantum inference.


AAAI manuscript deliverables:Generate LaTeX Results Table: Convert these benchmark metrics into an AAAI-formatted LaTeX table (\begin{table}...) for insertion into main.tex.Quantum Misclassification Analysis: Inspect the $2.6\%$ misclassified samples (the gap between $97.40\%$ and $100\%$) to determine if the errors stem from PCA information loss or entangling gate limitations.Draft Methodology & Results Text: Write the ArXiv/AAAI paper narrative covering the hybrid vision architecture: $\text{YOLOv8} \rightarrow \text{PCA Compression} \rightarrow \text{Angle Encoding} \rightarrow \text{VQC Measurement}$.

Generate LaTeX Results Table
Key Results SummaryQuantum Feature Compression: The 4-qubit Variational Quantum Classifier (VQC) achieves an F1-score of 0.9739 using 4 Principal Component Analysis (PCA) features compressed from the original 1024-dimensional YOLO crop vectors.Optimization Stability: Quantum loss steadily decreased from 3.8354 to 0.1277 over 15 epochs using PennyLane's Adam Optimizer.Classical Baseline: Both SVM and Random Forest achieved 100% accuracy on the full 1024-dimensional feature vectors with sub-millisecond inference times ($0.03\text{ ms}$ and $0.11\text{ ms}$).


#### Workflow Breakdown for Next Time
python benchmark_compare.py (Primary — Full Results & Metrics)

When to run: When you want to run end-to-end inference across your dataset, evaluate model performance, and generate complete benchmark results (Accuracy, F1-Score, Latency) comparing Classical (YOLO + SVM/Random Forest) vs. Quantum (VQC).

Prerequisite: Uses the existing trained best.pt model.

python classical_pipeline.py (Optional — Quick Crop Inspection)

When to run: When you only want to test single-image bounding box crop extraction and inspect individual YOLO vs. SVM crop predictions without running the full quantum benchmark harness.

Scripts You Can Skip
generate_labels.py: Skip. Only run if you add new raw images or modify the dataset structure.

python main.py yolo: Skip. Only run if you want to retrain the YOLO detection model from scratch (e.g., changing hyperparameters or epoch counts).
