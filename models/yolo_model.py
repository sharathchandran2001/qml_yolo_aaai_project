import os
import cv2
import numpy as np
from ultralytics import YOLO
from config import CROP_SIZE, DATA_DIR, BASE_DIR, YOLO_WEIGHTS_PATH

class YOLOCropExtractor:
    """
    Interface for Ultralytics YOLOv8 object detector and ground-truth crop extractor.
    Extracts regions of interest, resizes them to 32x32 grayscale vectors (1024-D),
    and returns dataset feature arrays.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            # Search for fine-tuned best.pt weights in runs/ directory
            runs_dir = os.path.join(BASE_DIR, "runs")
            fine_tuned_weights = None
            if os.path.exists(runs_dir):
                for root, _, files in os.walk(runs_dir):
                    if "best.pt" in files:
                        fine_tuned_weights = os.path.join(root, "best.pt")
                        break

            if fine_tuned_weights:
                model_path = fine_tuned_weights
                print(f"[+] Loaded fine-tuned YOLO model from: {fine_tuned_weights}")
            elif os.path.exists(YOLO_WEIGHTS_PATH):
                model_path = YOLO_WEIGHTS_PATH
                print(f"[+] Loaded base YOLO model from: {YOLO_WEIGHTS_PATH}")
            else:
                model_path = 'yolov8s.pt'

        self.model = YOLO(model_path)

    def extract_crops_from_labels(self, image_path: str):
        """
        Ground-truth extraction: Reads bounding boxes directly from matching
        YOLO annotation .txt label files if available.
        """
        img = cv2.imread(image_path)
        if img is None:
            return [], []

        h_img, w_img = img.shape[:2]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        label_candidates = [
            os.path.join(os.path.dirname(image_path), f"{base_name}.txt"),
            os.path.join(DATA_DIR, "labels", f"{base_name}.txt"),
            os.path.join(DATA_DIR, "raw", f"{base_name}.txt"),
            os.path.join(DATA_DIR, "images", f"{base_name}.txt"),
            os.path.join(DATA_DIR, "labels", "train", f"{base_name}.txt")
        ]

        label_path = None
        for candidate in label_candidates:
            if os.path.exists(candidate):
                label_path = candidate
                break

        if not label_path:
            return [], []

        crops = []
        class_ids = []

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])

            # Convert normalized YOLO coords (cx, cy, w, h) to pixel (x1, y1, x2, y2)
            x1 = int((cx - w / 2) * w_img)
            y1 = int((cy - h / 2) * h_img)
            x2 = int((cx + w / 2) * w_img)
            y2 = int((cy + h / 2) * h_img)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, CROP_SIZE, interpolation=cv2.INTER_AREA)
            flat_vec = resized.flatten().astype(np.float32) / 255.0

            crops.append(flat_vec)
            class_ids.append(cls_id)

        return crops, class_ids

    def extract_crop_from_image(self, image_path: str):
        """
        Extracts crops using YOLO inference, falling back to ground-truth label files.
        """
        img = cv2.imread(image_path)
        if img is None:
            return [], []

        results = self.model(img, verbose=False)
        crops = []
        class_ids = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, CROP_SIZE, interpolation=cv2.INTER_AREA)
                flat_vec = resized.flatten().astype(np.float32) / 255.0

                crops.append(flat_vec)
                class_ids.append(cls_id)

        # Fallback to label .txt file if YOLO inference returned 0 crops
        if len(crops) == 0:
            crops, class_ids = self.extract_crops_from_labels(image_path)

        return crops, class_ids

    def extract_all_crop_features(self, images_dir: str = None):
        """
        Scans dataset directory (excluding 'processed' output charts) and extracts feature vectors.
        """
        search_dir = images_dir if images_dir is not None else DATA_DIR
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = []

        if os.path.exists(search_dir):
            for root, _, files in os.walk(search_dir):
                # Ignore generated output plots / charts in processed/
                if "processed" in root:
                    continue
                for f in files:
                    if f.lower().endswith(valid_exts):
                        image_files.append(os.path.join(root, f))

        if len(image_files) == 0:
            print(f"[-] Warning: No dataset image files found in '{search_dir}'.")
            return np.array([]), np.array([])

        print(f"[+] Extracting crop features from {len(image_files)} image(s) under '{search_dir}'...")
        all_features = []
        all_labels = []

        for img_path in image_files:
            crops, cls_ids = self.extract_crop_from_image(img_path)
            all_features.extend(crops)
            all_labels.extend(cls_ids)

        return np.array(all_features), np.array(all_labels)