# MorphoCrack-Vision

> Deep Learning-based Structural Crack Detection System

## 專案簡介
MorphoCrack-Vision 是一個自動化的電腦視覺批次處理管線，專為土木與建築結構的健康監測（Structural Health Monitoring, SHM）所設計。本系統利用 YOLOv8 深度學習引擎，能夠快速讀取無人機或現場檢測之原始影像，進行裂縫的自動辨識、面積估算與風險分級，並產出視覺化之診斷報告。

## 核心功能
* **智慧化偵測**：基於 YOLOv8 實例分割 (Instance Segmentation) 技術，精準框選裂縫邊緣。
* **量化損傷評估**：自動計算影像中的損傷面積比 (Damage Ratio)。
* **風險警示系統**：依據受損比例自動進行風險分級 (SAFE / WARNING / CRITICAL)。
* **自動批次處理**：支援遞迴掃描資料夾，一鍵自動處理海量檢測照片，並依原資料夾結構進行輸出。

## 資料夾架構
本專案採用模組化設計，遵循標準資料科學專案結構：

```text
MorphoCrack-Vision/
├── data/
│   ├── raw/          # 原始輸入影像 (支援子資料夾結構)
│   └── processed/    # 系統自動生成之視覺化診斷結果
├── models/           # YOLO 模型權重檔存放區 (例如 best.pt)
├── preprocess.py     # 影像前處理模組 (OpenCV)
├── detection.py      # AI 偵測引擎模組 (Ultralytics YOLO)
├── analysis.py       # 損傷量化與邏輯分析模組
└── main.py           # 系統主控台與批次處理邏輯