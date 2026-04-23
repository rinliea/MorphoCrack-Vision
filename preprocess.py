import cv2
import numpy as np
import os

def prepare_input(image_path):
    """
    讀取影像並進行基本的格式確認。
    確保餵給 YOLOv8 引擎的圖像是正確的 RGB 格式。
    """
    if not os.path.exists(image_path):
        print(f"❌ 錯誤：找不到影像檔案 {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 錯誤：影像檔案毀損或無法讀取 {image_path}")
        return None
    
    # YOLO 預設處理 RGB 格式，而 OpenCV 讀取時是 BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb