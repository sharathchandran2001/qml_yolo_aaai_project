import base64
import requests

vertical_mermaid_code = """
flowchart TD
    classDef classical fill:#e3f2fd,stroke:#1565c0,stroke-width:1.5px,color:#0d47a1
    classDef quantum fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1.5px,color:#4a148c
    classDef eval fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px,color:#1b5e20

    SETUP["YOLOv8 Object Detector<br/>& Crop Extraction<br/>(1024-D Dense Vector)"]:::classical

    SETUP -->|"Path A: Uncompressed"| CLASSICAL["Classical Classifiers<br/>(SVM & Random Forest)"]:::classical

    SETUP -->|"Path B: 6-D Projection"| PCA["PCA Compression<br/>(1024-D → 6-D)"]:::quantum
    PCA --> VQC["6-Qubit VQC Circuit<br/>(RX Angle Map + L=3 Ansatz, 54 Params<br/>+ Pauli-Z ⟨Z₀⟩ Measurement)"]:::quantum

    CLASSICAL --> EVAL["Benchmarking Engine<br/>(Accuracy, F1, Latency)"]:::eval
    VQC --> EVAL
"""

def save_vertical_diagram(code: str, output_filename: str = "pipeline_diagram_1.png"):
    encoded_bytes = base64.b64encode(code.encode("utf-8"))
    encoded_string = encoded_bytes.decode("utf-8")
    
    url = f"https://mermaid.ink/img/{encoded_string}"
    
    print("[+] Fetching rendered vertical diagram from API...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(output_filename, "wb") as f:
            f.write(response.content)
        print(f"[✔] Successfully saved vertical diagram to: {output_filename}")
    else:
        print(f"[✘] Failed to retrieve image. Status code: {response.status_code}")

if __name__ == "__main__":
    save_vertical_diagram(vertical_mermaid_code)