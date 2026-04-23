from ultralytics import YOLO
import os

class YOLODetector:
    def __init__(self, weight_path='models/best.pt'):
        """
        初始化 YOLOv8 模型。
        請確保妳下載的權重檔已經放在 models/ 資料夾下。
        """
        self.weight_path = weight_path
        if not os.path.exists(self.weight_path):
            raise FileNotFoundError(f"⚠️ 找不到權重檔：請確認 {self.weight_path} 是否存在。")
        
        # 載入妳的模型
        self.model = YOLO(self.weight_path)

    def detect(self, img_rgb, confidence=0.25, iou=0.45):
        """
        執行裂縫偵測。
        可以透過調整 confidence 降低誤判，調整 iou 減少重複框選。
        """
        # save=False 代表不在這裡存圖，統一交給主程式處理
        results = self.model.predict(source=img_rgb, conf=confidence, iou=iou, save=False)
        return results[0]