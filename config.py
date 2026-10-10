from pathlib import Path

# ==============================================================================
# PROJECT DIRECTORY & FILE PATHS  (all pathlib.Path - compatible with os.path.join too)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_IMG_DIR = RAW_DATA_DIR                      # used by annotation_tool/capture.py
PROCESSED_DATA_DIR = DATA_DIR / "processed"

IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
YOLO_IMG_TRAIN = IMAGES_DIR / "train"           # used by annotation_tool/coordinate_tagger.py
YOLO_IMG_VAL = IMAGES_DIR / "val"
YOLO_LBL_TRAIN = LABELS_DIR / "train"
YOLO_LBL_VAL = LABELS_DIR / "val"
RESULTS_DIR = BASE_DIR / "results"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, YOLO_IMG_TRAIN, YOLO_IMG_VAL,
           YOLO_LBL_TRAIN, YOLO_LBL_VAL, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# YOLO Model Configuration
YOLO_WEIGHTS_PATH = BASE_DIR / "yolov8s.pt"
YOLO_RUNS_DIR = BASE_DIR / "runs"

# ==============================================================================
# FEATURE EXTRACTION & IMAGE PREPROCESSING
# ==============================================================================
CROP_SIZE = (32, 32)          # Standardized bounding box crop dimension
FEATURE_DIM = 1024            # Flattened feature dimension (32 * 32)

# ==============================================================================
# QUANTUM MACHINE LEARNING (QML) HYPERPARAMETERS  (defaults; qml_benchmark.py CLI overrides)
# ==============================================================================
DEFAULT_DIMS = [6, 12, 24, 48]  # PCA feature dimensions swept by the benchmark
NUM_LAYERS = 3                  # StronglyEntanglingLayers depth
LEARNING_RATE = 0.1             # Adam optimizer learning rate
EPOCHS = 50                     # Training epochs for PennyLane VQC
QML_MAX_QUBITS = 12             # dims above this pack several features per qubit
NUM_QUBITS = 6                  # legacy name, kept for older scripts

# Class Mappings (Binary)
CLASS_NAMES = ["calculator_display", "button_node"]
