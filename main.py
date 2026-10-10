"""
main.py - pipeline orchestration.

  python main.py capture                       # grab a screenshot
  python main.py yolo [epochs]                 # fine-tune YOLOv8s on data/images
  python main.py benchmark --dims 6 12 24 48   # dimension sweep (all qml_benchmark.py options work)
  python main.py report                        # tables + charts from every run so far
"""
import sys

from config import BASE_DIR, DATA_DIR, YOLO_RUNS_DIR, YOLO_WEIGHTS_PATH


def run_screen_capture():
    from annotation_tool.capture import capture_screen_region
    save_path, _ = capture_screen_region()
    print(f"Captured screen saved to: {save_path}")


def run_yolo_pipeline(epochs=5):
    from ultralytics import YOLO
    train_dir = DATA_DIR / "images" / "train"
    imgs = [p for p in train_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not imgs:
        print("[!] No training images in data/images/train/ - add images + labels first.")
        return
    print(f"Found {len(imgs)} training images. Starting YOLOv8 training...")
    weights = YOLO_WEIGHTS_PATH if YOLO_WEIGHTS_PATH.exists() else "yolov8s.pt"
    YOLO(str(weights)).train(data=str(DATA_DIR / "dataset.yaml"), epochs=epochs, imgsz=640,
                             project=str(YOLO_RUNS_DIR), name="yolov8s_aaai_experiment", exist_ok=True)


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if mode == "capture":
        run_screen_capture()
    elif mode == "yolo":
        run_yolo_pipeline(int(rest[0]) if rest else 5)
    elif mode in ("benchmark", "ml"):
        from qml_benchmark import main as bench
        bench(rest)
    elif mode == "report":
        from generate_report import main as report
        report(rest)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
