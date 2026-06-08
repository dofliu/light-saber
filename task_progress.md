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

### 目前進行中
- 等待 #1 合併；本次 `--max-hands` 變更以 stacked PR 方式接在 #1 後面

### 下一個候選項目
- 將 `lightsaber_mvp.py` 的 Tunables 抽成 `config` 區塊，便於快速調參
- 增加 `--display-size` 參數，提升展示現場調整彈性
