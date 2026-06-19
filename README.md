# Lightsaber MVP

即時手勢辨識光劍互動 Demo（OpenCV + MediaPipe）。

## 環境需求

- Python 3.10 以上（建議 3.11）
- 可用的 webcam
- Windows 11（其他平台理論可行，但未在本專案驗證）

## 安裝

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 執行

```bash
python lightsaber_mvp.py
```

可選參數：

```bash
python lightsaber_mvp.py --camera-index 1 --no-mirror --max-hands 2 --display-size 1280x720 --process-scale 0.75 --model-complexity 1
```

## 操作說明

- 握拳持續 0.5 秒：點燃光劍
- 張手持續 0.3 秒：收回光劍
- `F`：切換 fullscreen
- `M`：切換 mirror
- `D`：切換 landmark debug overlay
- `1 / 2`：縮短 / 加長 blade
- `ESC / Q`：離開

## 專案結構

- `lightsaber_mvp.py`：主程式
- `requirements.txt`：Python 依賴
- `.gitignore`：忽略本機與暫存檔
- `tests/`：自動化測試
- `task_progress.md`：每次單一任務推進記錄

## 已知限制

- 多人同框時，手部 slot 配對仍可能在快速遮擋下短暫切換。
- 聲音輸出依賴 `pygame`，若音訊裝置初始化失敗，相關效果會受限。
