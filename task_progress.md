# Lightsaber 專案進度

## 2026-05-23

### 已完成
- 建立專案基礎維護檔案：`README.md`、`requirements.txt`、`.gitignore`
- 補上啟動方式、快捷鍵說明與已知限制，讓專案可快速交接
- 增加 CLI 參數：`--camera-index`、`--no-mirror`，提升不同現場設備啟動成功率

## 2026-06-07

### 已完成
- 補上 CLI 參數 smoke test，驗證預設 camera/mirror 行為與自訂參數解析
- 修正 `README.md` 與 `task_progress.md` 的繁中內容，避免文件在後續維護中不可讀

## 2026-06-08

### 已完成
- 增加 CLI 參數：`--max-hands`，讓展示現場可限制手部追蹤數量
- 補上 `--max-hands` 測試，涵蓋預設值、自訂值與非法值

## 2026-06-18

### 已完成
- 增加 CLI 參數：`--display-size WIDTHxHEIGHT`，讓展示現場可調整 OpenCV 視窗初始大小
- 補上 `--display-size` 測試，涵蓋預設值、自訂值、格式錯誤與非正整數

## 2026-06-19

### 已完成
- 增加 CLI 參數：`--process-scale`，便於展示現場在效能與追蹤精度間調整
- 補上 `--process-scale` 測試，涵蓋預設值、自訂值、零值與超出上限
- 增加 CLI 參數：`--model-complexity`，便於不同設備調整 MediaPipe 模型負載
- 補上 `--model-complexity` 測試，限制值只能是 `0` 或 `1`

### 目前進行中
- 等待 #1、#2、#3 與 #4 合併；本次 `--model-complexity` 變更以 stacked PR 方式接在 #4 後面

### 下一個候選項目
- 將 `lightsaber_mvp.py` 的 Tunables 抽成 `config` 區塊，便於快速調參
- 增加 `--min-detection-confidence` 與 `--min-tracking-confidence` 參數，讓展示現場可調整追蹤穩定度
