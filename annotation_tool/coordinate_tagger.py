import cv2
import shutil
from pathlib import Path
from config import (
    YOLO_IMG_TRAIN, YOLO_LBL_TRAIN,
    YOLO_IMG_VAL, YOLO_LBL_VAL
)

def convert_to_yolo(size, box):
    """
    Converts (x1, y1, x2, y2) pixel bounding box to YOLO normalized format:
    [center_x, center_y, width, height]
    """
    img_w, img_h = size
    x1, y1, x2, y2 = box

    # Calculate center, width, height
    cx = (x1 + x2) / (2.0 * img_w)
    cy = (y1 + y2) / (2.0 * img_h)
    w = (x2 - x1) / float(img_w)
    h = (y2 - y1) / float(img_h)

    return (cx, cy, w, h)


def create_yolo_label(image_path, annotations, is_val=False):
    """
    Processes an image and its raw pixel annotations.
    """
    image_path = Path(image_path).resolve()
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    h, w, _ = img.shape

    # Define destination paths
    dst_img_dir = YOLO_IMG_VAL if is_val else YOLO_IMG_TRAIN
    dst_lbl_dir = YOLO_LBL_VAL if is_val else YOLO_LBL_TRAIN

    target_img_path = (dst_img_dir / image_path.name).resolve()

    # Copy image only if target path is different from source path
    if image_path != target_img_path:
        shutil.copy(image_path, target_img_path)
        print(f"[+] Copied image to: {target_img_path}")
    else:
        print(f"[+] Image already located at target directory: {target_img_path}")

    # Write YOLO label file (.txt)
    txt_filename = image_path.stem + ".txt"
    target_txt_path = dst_lbl_dir / txt_filename

    with open(target_txt_path, "w") as f:
        for ann in annotations:
            cls_id, x1, y1, x2, y2 = ann
            cx, cy, box_w, box_h = convert_to_yolo((w, h), (x1, y1, x2, y2))
            # Format: class_id center_x center_y width height
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}\n")

    print(f"[+] Label written to: {target_txt_path}")