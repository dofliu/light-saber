# Lightsaber MVP

以 Python、OpenCV 與 MediaPipe Hands 製作的多人即時互動光劍遊戲。系統從 webcam 追蹤手部姿態，以握拳啟動、張手收回光劍，並提供 Arcade 目標挑戰、計分、Combo、光暈、殘影、碰撞火花與程式合成音效。

## 主要功能

- 最多同時追蹤 4 隻手，為每隻手維持獨立光劍狀態與顏色。
- 握拳持續 0.5 秒啟動光劍；張手持續 0.3 秒收回光劍。
- 使用 One Euro Filter 降低 landmark 抖動，並在短暫漏偵測時進行 velocity glide。
- 支援光劍伸縮、光暈、揮動殘影、揮劍音效、光劍交擊火花與碰撞音效。
- 可透過 CLI 調整 camera、手部數量、視窗尺寸與 MediaPipe 效能參數。
- Arcade mode 提供 3 秒倒數、限時目標、揮劍速度門檻、命中評價、Combo 與回合結算。

## 執行環境

- Windows 11（目前主要測試平台）
- Python 3.10 或 3.11
- Webcam

## 安裝

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 啟動

使用預設設定：

```powershell
python lightsaber_mvp.py
```

自訂 camera 與處理參數：

```powershell
python lightsaber_mvp.py --camera-index 1 --no-mirror --max-hands 2 --display-size 1280x720 --process-scale 0.75 --model-complexity 1
```

查看完整參數：

```powershell
python lightsaber_mvp.py --help
```

## CLI 參數

| 參數 | 預設值 | 說明 |
| --- | --- | --- |
| `--camera-index` | `0` | OpenCV 使用的 webcam index。 |
| `--no-mirror` | 關閉 | 停用預覽畫面的水平鏡像。 |
| `--max-hands` | `4` | 同時追蹤的手部數量，必須大於 0。 |
| `--display-size` | `1600x900` | 初始視窗大小，格式為 `WIDTHxHEIGHT`。 |
| `--process-scale` | `0.5` | MediaPipe 處理前的影像縮放比例，範圍為 `0 < value <= 1`。 |
| `--model-complexity` | `0` | MediaPipe Hands 模型複雜度，可選 `0` 或 `1`。 |
| `--game-mode` | `arcade` | 啟動時使用 `arcade` 或 `free` mode。 |
| `--round-seconds` | `60` | Arcade 回合秒數，必須大於 0。 |
| `--game-seed` | 無 | 固定目標亂數，供測試與重現 demo 使用。 |

## 操作方式

| 操作 | 功能 |
| --- | --- |
| 握拳 0.5 秒 | 啟動光劍 |
| 張手 0.3 秒 | 收回光劍 |
| `F` | 切換 fullscreen |
| `M` | 切換鏡像 |
| `D` | 顯示或隱藏 hand landmarks |
| `Space` | 開始 Arcade 回合／重新挑戰 |
| `G` | 切換 Arcade 與自由揮劍模式 |
| `R` | 重設目前回合 |
| `1` / `2` | 縮短／加長光劍 |
| `Esc` / `Q` | 結束程式 |

## 測試

```powershell
python -m unittest discover -s tests -v
python -m py_compile lightsaber_mvp.py tests/test_cli_args.py
```

目前 automated tests 涵蓋 CLI 驗證、遊戲狀態轉換、目標生命週期、線段命中、揮劍速度門檻、計分與 Combo。Camera、手勢辨識、畫面與音效仍需以 webcam 進行手動 smoke test。

## 專案結構

```text
.
├── lightsaber_mvp.py       # 主程式、手勢判定、狀態管理、特效與音效
├── requirements.txt        # Python dependencies
├── tests/
│   ├── test_arcade_game.py # Arcade gameplay logic tests
│   └── test_cli_args.py    # CLI argument tests
├── ROADMAP.md              # 後續開發階段與驗收條件
└── task_progress.md        # 歷史開發紀錄
```

## 技術流程

```text
Webcam frame
  -> resize / mirror
  -> MediaPipe Hands
  -> gesture classification
  -> hand-slot assignment and smoothing
  -> lightsaber state transition
  -> visual effects and synthesized audio
  -> OpenCV display
```

## 已知限制

- 手勢判定目前以幾何規則為主，會受遮擋、光線與手掌角度影響。
- 光劍碰撞採 2D 線段交點判定，不代表真實 3D 空間碰撞。
- 主要程式集中在單一 Python 檔案，後續擴充與 isolated testing 成本較高。
- 尚未建立 camera-independent 的錄影回放測試與效能 benchmark。

後續開發規劃與優先順序請參考 [ROADMAP.md](ROADMAP.md)。
