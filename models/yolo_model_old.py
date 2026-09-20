from ultralytics import YOLO
from config import DATA_DIR, BASE_DIR

class YoloDetector:
    def __init__(self, model_version="yolov8s.pt"):
        # Load YOLOv8 Small pretrained weights
        self.model = YOLO(model_version)

    def train_model(self, epochs=50, imgsz=640):
        dataset_yaml = DATA_DIR / "dataset.yaml"
        results = self.model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=imgsz,
            project=str(BASE_DIR / "runs"),
            name="yolov8s_aaai_experiment"
        )
        return results

    def run_inference(self, image_path):
        results = self.model(image_path)
        return results