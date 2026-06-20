# Lightsaber 開發紀錄

## 2026-05-23

- 建立 `README.md`、`requirements.txt` 與 `.gitignore`。
- 建立多人 webcam 光劍 MVP。
- 新增 `--camera-index` 與 `--no-mirror` CLI 參數。

## 2026-06-07

- 新增 camera 與 mirror CLI smoke tests。

## 2026-06-08

- 新增 `--max-hands` 參數與輸入驗證。

## 2026-06-18

- 新增 `--display-size WIDTHxHEIGHT` 參數與輸入驗證。

## 2026-06-19

- 新增 `--process-scale` 與 `--model-complexity` 參數。
- 補齊相關 CLI tests。

## 2026-06-21

- 將所有 stacked feature branches fast-forward 整併至 `main`，刪除其餘 local 與 remote branches。
- 重寫專案 README，新增後續開發 Roadmap。
- 新增 Arcade mode、限時目標、Score、Combo、命中評價與回合結算。
- 新增遊戲核心 unit tests，讓 gameplay logic 可脫離 webcam 驗證。
- 新增方向目標、揮劍方向評分與 `WRONG WAY` 回饋。
- 新增 `easy / normal / hard` difficulty presets 與相關 tests。
- 新增敵方雷射、光劍 Parry、Shield lives 與攻防循環 tests。
- 新增 Pause time freeze 與 persistent high score。
