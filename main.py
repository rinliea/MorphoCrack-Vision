from preprocess import prepare_input
from detection import YOLODetector
from analysis import analyze_yolo_results
import os

def run_batch_diagnosis(weight_file='best.pt'):
    # 1. 設定基礎路徑
    raw_dir = os.path.join('data', 'raw')
    processed_dir = os.path.join('data', 'processed')
    weight_path = os.path.join('models', weight_file)
    
    if not os.path.exists(raw_dir):
        print(f"❌ 錯誤：找不到 {raw_dir} 資料夾")
        return

    # 2. 升級版雷達：掃描所有子資料夾 (包含 positive 和 negative)
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    image_tasks = []
    
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(valid_ext):
                # 取得檔案的完整路徑
                full_path = os.path.join(root, file)
                # 取得相對路徑 (例如: positive/test.jpg)
                rel_path = os.path.relpath(full_path, raw_dir)
                image_tasks.append((full_path, rel_path))

    if len(image_tasks) == 0:
        print(f"⚠️ 警告：在 {raw_dir} 及其子資料夾內找不到任何圖片。")
        return

    print("="*50)
    print(f"🚀 啟動 MorphoCrack-Vision (雙資料夾批次處理模式)")
    print(f"📂 共掃描到 {len(image_tasks)} 張圖片準備診斷")
    print("="*50 + "\n")

    try:
        # 3. 載入 YOLO 模型
        detector = YOLODetector(weight_path=weight_path) 
        
        # 4. 開始處理每一張圖
        for idx, (full_path, rel_path) in enumerate(image_tasks, 1):
            # 建立對應的輸出路徑 (自動生成 positive / negative 資料夾)
            output_path = os.path.join(processed_dir, f"result_{rel_path}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            print(f"🔄 處理中 ({idx}/{len(image_tasks)}): [{rel_path}] ...")
            
            # 讀取影像
            img_rgb = prepare_input(full_path)
            if img_rgb is None: 
                continue

            # 執行偵測與分析
            detection_results = detector.detect(img_rgb, confidence=0.3)
            report = analyze_yolo_results(detection_results)

            # 儲存圖片
            detection_results.save(filename=output_path)
            
            # 列印單圖簡報
            print(f"   ↳ 裂縫數: {report['crack_count']} | 風險: {report['risk_level']} | 損傷比: {report['ratio_percentage']}")

        print("\n" + "="*50)
        print(f"✅ 批次處理完成！結果已分類存入 {processed_dir}")
        print("="*50)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"❌ 系統執行時發生未預期的錯誤: {e}")

if __name__ == "__main__":
    run_batch_diagnosis(weight_file="best.pt")