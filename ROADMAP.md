# Lightsaber MVP 後續工作規劃

## 目標

將目前可運作的單檔 MVP，逐步整理為可測試、可調校、可展示及可延伸研究的即時互動系統。各階段依相依性排序，完成前一階段後再進入下一階段。

## Phase 1：可靠性與可重現性

- 將 `min_detection_confidence` 與 `min_tracking_confidence` 加入 CLI，並補齊範圍驗證 tests。
- 新增 camera 啟動失敗、讀取中斷與 MediaPipe 初始化失敗的可讀錯誤訊息。
- 建立固定 webcam smoke-test checklist，記錄單手、雙手、四手、遮擋及低光源情境。
- 固定 dependency versions，確認 Python 3.10 與 3.11 的 clean-install 結果。

完成條件：automated tests 全數通過，且 smoke-test checklist 無阻斷性問題。

## Phase 2：模組化與測試

- 將 CLI/config、gesture detection、hand tracking、rendering、audio 拆成獨立 modules。
- 將目前散落的 tunables 集中為 config dataclass，避免 runtime 設定與常數重複。
- 為 hand-slot assignment、gesture state transition、collision detection 增加 unit tests。
- 建立不依賴 webcam 的 recorded-frame 或 synthetic-landmark regression tests。

完成條件：核心邏輯可在不開啟 camera 與視窗的情況下執行測試。

## Phase 3：效能量測與調校

- 新增 FPS、MediaPipe inference time、render time 與 dropped-frame 指標。
- 建立 `process-scale`、`model-complexity`、`max-hands` 的 benchmark matrix。
- 針對 RTX 4080／CPU 執行環境記錄 baseline，不將估算值當成實測結果。
- 依量測結果調整 frame processing 與 effect rendering bottlenecks。

完成條件：產出可重現 benchmark script、測試條件與實測報告。

## Phase 4：互動與展示品質

- 增加 calibration 畫面與目前 gesture state 的視覺回饋。
- 支援光劍顏色、音量、靈敏度與劍長設定檔。
- 改善手部交叉、短暫遮擋與 participant 進出畫面時的 slot identity 穩定性。
- 規劃 demo mode、操作說明 overlay 與一鍵啟動 script。

完成條件：一般使用者可在無口頭指導下完成啟動、揮劍、交擊與退出流程。

## Phase 5：研究延伸

- 比較幾何規則、SVM 與 landmark sequence model 的 gesture recognition 表現。
- 建立包含不同使用者、光線、背景與遮擋條件的評估資料集。
- 定義 accuracy、latency、jitter、false activation rate 與 recovery time 指標。
- 評估 2D interaction、depth camera 或 multi-view 方案的研究價值。

完成條件：研究問題、資料收集 protocol、evaluation metrics 與 baseline 均可重現。

## 建議下一個 Sprint

1. 新增 detection/tracking confidence CLI 與 tests。
2. 建立 webcam smoke-test checklist。
3. 拆出 config 與 gesture state transition。
4. 為 state transition 補 unit tests。
5. 建立第一版 FPS／inference-time benchmark。
