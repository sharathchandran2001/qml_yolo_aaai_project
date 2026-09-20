import os
import cv2
from pathlib import Path
from config import DATA_DIR

def convert_to_yolo_format(box, img_w, img_h):
    """
    Converts pixel bounding box (cls_id, x1, y1, x2, y2) to 
    normalized YOLO format: (cls_id, cx, cy, w, h).
    """
    cls_id, x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    
    norm_cx = max(0.0, min(1.0, cx / img_w))
    norm_cy = max(0.0, min(1.0, cy / img_h))
    norm_w = max(0.0, min(1.0, bw / img_w))
    norm_h = max(0.0, min(1.0, bh / img_h))
    
    return f"{cls_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n"

def process_folder(image_folder: Path, label_folder: Path):
    """
    Scans image folder, calculates bounding boxes, and creates matching .txt label files.
    """
    label_folder.mkdir(parents=True, exist_ok=True)
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    img_files = [f for f in image_folder.glob("*") if f.suffix.lower() in valid_exts]

    if not img_files:
        print(f"[-] No image files found in '{image_folder}'")
        return

    for img_path in sorted(img_files):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[!] Warning: Could not read {img_path.name}")
            continue

        h, w, _ = img.shape
        display_y2 = int(h * 0.35)

        # Standard bounding box annotations per image: (class_id, x1, y1, x2, y2)
        # Class 0: calculator_display | Class 1: button_node
        annotations = [
            # Display area
            (0, 10, 40, w - 10, display_y2),
            # Button nodes
            (1, int(w * 0.02), int(h * 0.55), int(w * 0.24), int(h * 0.63)),
            (1, int(w * 0.26), int(h * 0.55), int(w * 0.48), int(h * 0.63)),
            (1, int(w * 0.76), int(h * 0.88), int(w * 0.98), int(h * 0.98))
        ]

        label_path = label_folder / f"{img_path.stem}.txt"
        with open(label_path, "w") as f:
            for box in annotations:
                f.write(convert_to_yolo_format(box, w, h))

        print(f"[+] Created label: {label_path.relative_to(Path(DATA_DIR).parent)}")

def main():
    data_path = Path(DATA_DIR)
    
    train_img_dir = data_path / "images" / "train"
    train_lbl_dir = data_path / "labels" / "train"
    val_img_dir = data_path / "images" / "val"
    val_lbl_dir = data_path / "labels" / "val"

    print("[+] Generating YOLO annotations for training set...")
    process_folder(train_img_dir, train_lbl_dir)

    print("\n[+] Generating YOLO annotations for validation set...")
    process_folder(val_img_dir, val_lbl_dir)

    print("\n[+] Label generation complete.")

if __name__ == "__main__":
    main()