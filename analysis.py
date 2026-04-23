import numpy as np

def analyze_yolo_results(results):
    """
    解析 YOLOv8 的輸出結果，進行形態學量化評估。
    """
    analysis_report = {
        "crack_count": 0,
        "total_area_px": 0,
        "risk_level": "LOW (安全)",
        "ratio_percentage": "0.00%"
    }

    if results.boxes is None or len(results.boxes) == 0:
        return analysis_report

    # 記錄裂縫數量
    analysis_report["crack_count"] = len(results.boxes)

    # 如果妳使用的是分割模型 (Segmentation)，會產生 masks
    if hasattr(results, 'masks') and results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        # (修正後的寫法：確保分子與分母的尺寸基準一致)
        total_pixels = masks.shape[1] * masks.shape[2]
        
        # 聯集所有裂縫遮罩，計算總受損像素
        combined_mask = np.sum(masks, axis=0)
        crack_pixel_count = np.count_nonzero(combined_mask > 0)
        
        analysis_report["total_area_px"] = crack_pixel_count
        ratio = (crack_pixel_count / total_pixels) * 100
        analysis_report["ratio_percentage"] = f"{ratio:.4f}%"
        
        # 結構風險判定邏輯 (面試時可說明這是動態參數)
        if ratio > 1.0: 
            analysis_report["risk_level"] = "CRITICAL (危險)"
        elif ratio > 0.3: 
            analysis_report["risk_level"] = "MEDIUM (需注意)"

    return analysis_report