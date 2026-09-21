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

# Subdirectories for YOLO Training & Validation Data
YOLO_IMG_TRAIN = os.path.join(IMAGES_DIR, "train")
YOLO_LABEL_TRAIN = os.path.join(LABELS_DIR, "train")
YOLO_LBL_TRAIN = YOLO_LABEL_TRAIN

YOLO_IMG_VAL = os.path.join(IMAGES_DIR, "val")
YOLO_LABEL_VAL = os.path.join(LABELS_DIR, "val")
YOLO_LBL_VAL = YOLO_LABEL_VAL

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
NUM_QUBITS = 6                # Increased to 6 Qubits (6-D PCA space)
NUM_LAYERS = 3                # StronglyEntanglingLayers depth
LEARNING_RATE = 0.05          # Learning rate for VQC optimizer
EPOCHS = 50                   # Extended epochs for convergence

# Class Mappings (Binary)
CLASS_NAMES = ["calculator_display", "button_node"]