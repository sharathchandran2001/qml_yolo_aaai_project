import base64
import requests

# Updated 4-Stage Workflow Mermaid Code
mermaid_code = """
flowchart TD
    classDef humanNode fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100
    classDef classicalNode fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    classDef quantumNode fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    classDef evalNode fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20

    subgraph DataPrep ["Stage 1: Human-in-the-Loop & Object Detection Setup"]
        direction TB
        A["Human-in-the-Loop"]:::humanNode -->|"Upload Raw Dataset"| B["Calculator Images"]:::humanNode
        B -->|"Bounding Box Labeling"| C["Manual Annotation<br/>(generate_labels.py)"]:::humanNode
        C -->|"Train Object Detector"| D["YOLOv8 Training<br/>(main.py yolo → best.pt)"]:::classicalNode
    end

    subgraph FeatureExtraction ["Stage 2: Spatial Feature Extraction & Branching"]
        direction TB
        D --> E["YOLOv8 Spatial Detection<br/>(Bounding Box Inference)"]:::classicalNode
        E --> F["UI Bounding Box Crop<br/>(32 × 32 Pixel Resize)"]:::classicalNode
        F --> G["Uncompressed Feature Vector<br/>(1024-Dimensional Vector)"]:::classicalNode
    end

    subgraph Classification ["Stage 3: Dual-Path Machine Learning"]
        direction TB
        G -->|"Path A: Uncompressed 1024-D"| H["Classical ML Classifiers<br/>(SVM & Random Forest)"]:::classicalNode
        G -->|"Path B: Orthogonal Projection"| I["PCA Dimensionality Reduction<br/>(1024-D → 4-D Compression)"]:::quantumNode

        I -->|"Compressed 4-D Vector"| J["4-Qubit Angle Encoding<br/>(Single-Qubit RX Gates)"]:::quantumNode
        J --> K["Strongly Entangling Ansatz<br/>(L=3 Layers, 36 Trainable Params)"]:::quantumNode
        K --> L["Pauli-Z Expectation Measurement<br/>(Qubit 0 State Measurement ⟨Z₀⟩)"]:::quantumNode
        L --> M["Variational Quantum Classifier<br/>(QML Prediction)"]:::quantumNode

        H --> N["Classical Class Predictions"]:::classicalNode
        M --> O["Quantum Class Predictions"]:::quantumNode
    end

    subgraph Evaluation ["Stage 4: Performance Evaluation"]
        direction TB
        N --> P["Comparative Benchmark Engine<br/>(benchmark_compare.py)"]:::evalNode
        O --> P
        P --> Q["Performance Metrics Output<br/>(Accuracy, F1-Score, Latency Report)"]:::evalNode
    end

    style DataPrep fill:#fffaf5,stroke:#e65100,stroke-width:1.5px
    style FeatureExtraction fill:#f4f8fb,stroke:#1565c0,stroke-width:1.5px
    style Classification fill:#fcf8fc,stroke:#7b1fa2,stroke-width:1.5px
    style Evaluation fill:#f1f8f1,stroke:#2e7d32,stroke-width:1.5px
"""

def save_mermaid_png(code: str, output_filename: str = "updated_pipeline_workflow.png"):
    # Base64 encode the string for the endpoint
    encoded_bytes = base64.b64encode(code.encode("utf-8"))
    encoded_string = encoded_bytes.decode("utf-8")
    
    url = f"https://mermaid.ink/img/{encoded_string}"
    
    print(f"[+] Fetching rendered diagram from API...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(output_filename, "wb") as f:
            f.write(response.content)
        print(f"[✔] Successfully saved diagram to: {output_filename}")
    else:
        print(f"[✘] Failed to retrieve image. Status code: {response.status_code}")

if __name__ == "__main__":
    save_mermaid_png(mermaid_code)