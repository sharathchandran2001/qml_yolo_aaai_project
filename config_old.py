import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_IMG_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# YOLO dataset structure
YOLO_IMG_TRAIN = DATA_DIR / "images" / "train"
YOLO_IMG_VAL = DATA_DIR / "images" / "val"
YOLO_LBL_TRAIN = DATA_DIR / "labels" / "train"
YOLO_LBL_VAL = DATA_DIR / "labels" / "val"

CLASSES = ["calculator_display", "button_node"]
NUM_CLASSES = len(CLASSES)

# Ensure required directories exist
for path in [RAW_IMG_DIR, PROCESSED_DIR, YOLO_IMG_TRAIN, YOLO_IMG_VAL, YOLO_LBL_TRAIN, YOLO_LBL_VAL]:
    path.mkdir(parents=True, exist_ok=True)