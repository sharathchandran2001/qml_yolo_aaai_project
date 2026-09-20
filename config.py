import os

# ==============================================================================
# PROJECT DIRECTORY & FILE PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

# YOLO Model Configuration
YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "yolov8s.pt")
YOLO_RUNS_DIR = os.path.join(BASE_DIR, "runs")

# ==============================================================================
# FEATURE EXTRACTION & IMAGE PREPROCESSING
# ==============================================================================
CROP_SIZE = (32, 32)          # Standardized bounding box crop dimension
FEATURE_DIM = 1024            # Flattened feature dimension (32 * 32)

# ==============================================================================
# QUANTUM MACHINE LEARNING (QML) HYPERPARAMETERS
# ==============================================================================
NUM_QUBITS = 4                # Number of qubits (4-D input via PCA)
NUM_LAYERS = 3                # StronglyEntanglingLayers depth
LEARNING_RATE = 0.1           # Adam optimizer learning rate
EPOCHS = 20                   # Training epochs for PennyLane VQC

# Class Mappings (Binary)
CLASS_NAMES = ["calculator_display", "button_node"]