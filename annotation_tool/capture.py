import cv2
import time
import numpy as np
from PIL import ImageGrab
from config import RAW_IMG_DIR

def capture_screen_region(bbox=None, filename_prefix="calc_capture"):
    """
    Captures a screen region (e.g., Windows Calculator).
    bbox format: (left, top, right, bottom). If None, captures full screen.
    """
    print("Capturing screen in 3 seconds... Switch to target window.")
    time.sleep(3)
    
    screenshot = ImageGrab.grab(bbox=bbox)
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    timestamp = int(time.time())
    save_path = RAW_IMG_DIR / f"{filename_prefix}_{timestamp}.png"
    cv2.imwrite(str(save_path), img)
    print(f"[+] Saved screenshot to: {save_path}")
    return save_path, img

if __name__ == "__main__":
    capture_screen_region()