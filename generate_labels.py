import cv2
from pathlib import Path
from annotation_tool.coordinate_tagger import create_yolo_label

def main():
    # Path to your captured image
    image_path = Path("data/images/train/calc_sample_01.png")
    
    if not image_path.exists():
        print(f"[!] Error: Could not find {image_path}")
        return

    # Read image to get dynamic dimensions for accurate normalization
    img = cv2.imread(str(image_path))
    h, w, _ = img.shape
    print(f"Loaded image with dimensions: Width={w}, Height={h}")

    # Estimated Pixel Coordinates: (class_id, x1, y1, x2, y2)
    # Class 0: calculator_display | Class 1: button_node
    # Note: These are estimated based on typical Windows 11 calculator proportions.
    
    display_y2 = int(h * 0.35)  # Display takes up roughly the top 35%
    
    annotations = [
        # Calculator Display (Class 0)
        (0, 10, 40, w - 10, display_y2),
        
        # Button '7' (Class 1) - Approx middle-left
        (1, int(w * 0.02), int(h * 0.55), int(w * 0.24), int(h * 0.63)),
        
        # Button '8' (Class 1) - Next to '7'
        (1, int(w * 0.26), int(h * 0.55), int(w * 0.48), int(h * 0.63)),
        
        # Button '=' (Class 1) - Bottom right orange button
        (1, int(w * 0.76), int(h * 0.88), int(w * 0.98), int(h * 0.98))
    ]

    # Generate the YOLO label file
    create_yolo_label(image_path, annotations, is_val=False)
    print("\n[+] Label generation complete. Check data/labels/train/")

if __name__ == "__main__":
    main()