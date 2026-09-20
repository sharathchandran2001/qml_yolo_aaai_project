import cv2
import numpy as np
from pathlib import Path
from models.yolo_model import YoloDetector
from models.classical_ml import ClassicalClassifier

def extract_features_from_crop(crop_img, target_size=(32, 32)):
    """Resizes bounding box crop to 32x32 grayscale and flattens into a normalized feature vector."""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    return resized.flatten() / 255.0  # Normalize pixel values to [0, 1]

def extract_yolo_crop_features(image_path, yolo_detector):
    """Runs trained YOLO model, crops bounding boxes, and extracts feature vectors."""
    results = yolo_detector.run_inference(str(image_path))
    features = []
    boxes_coords = []
    class_ids = []

    img = cv2.imread(str(image_path))
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
                
            feat = extract_features_from_crop(crop)
            features.append(feat)
            boxes_coords.append((x1, y1, x2, y2))
            class_ids.append(cls_id)

    return np.array(features), boxes_coords, class_ids

def run_classical_ml_experiment():
    # Load custom trained YOLO weights
    weights_path = Path("runs/yolov8s_aaai_experiment/weights/best.pt")
    if not weights_path.exists():
        print("[!] Custom weights not found, using default yolov8s.pt")
        weights_path = "yolov8s.pt"

    print(f"[+] Loading trained YOLO model from: {weights_path}")
    yolo = YoloDetector(model_version=str(weights_path))

    # Input image path
    test_img_path = Path("data/images/train/calc_sample_01.png")
    
    print(f"[+] Running YOLO inference on: {test_img_path}")
    feats, boxes, labels = extract_yolo_crop_features(test_img_path, yolo)

    if len(feats) == 0:
        print("[!] No bounding boxes detected by YOLO.")
        return

    print(f"[+] Extracted {len(feats)} detection crops (Feature vector dim: {feats.shape[1]}).")

    # Train Classical Classifier (SVM) on extracted feature crops
    X_train = np.vstack([feats, feats + np.random.normal(0, 0.02, feats.shape)])
    y_train = np.tile(labels, 2)

    clf = ClassicalClassifier(model_type="svm")
    clf.train(X_train, y_train)

    # Perform inference
    predictions = clf.predict(feats)
    
    print("\n================ Pipeline Execution Results ================")
    class_map = {0: "calculator_display", 1: "button_node"}
    for idx, (box, orig_cls, pred_cls) in enumerate(zip(boxes, labels, predictions)):
        yolo_label = class_map.get(orig_cls, str(orig_cls))
        svm_label = class_map.get(pred_cls, str(pred_cls))
        print(f" Crop #{idx+1} | BBox: {box} | YOLO Detected: {yolo_label} | SVM Predicted: {svm_label}")

if __name__ == "__main__":
    run_classical_ml_experiment()