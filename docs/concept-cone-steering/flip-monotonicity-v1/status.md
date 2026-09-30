# flip-monotonicity-v1 status

**結論（2026-09-30）：**以生成決策的 flip-rate 曲線重算，**判準只在 GLM 的 buy→sell 單一方向成立**；Qwen、Gemma、GPT-OSS 都不成立。而且成立的那一處並不表示 cone4 比 DIM 平滑：cone4 與 DIM 的 monotone-step fraction 都是 1.0（平手），勝出的原因是 `dim_orth_rand4` 在高劑量崩壞。整體看，cone4 沒有一致地比 DIM 平滑，與 margin 版 C8 的「不支持」方向一致，但兩者不是同一個指標。協議見 [proposal.md](proposal.md)。

## 結果（ITT on-target flip rate 曲線，相鄰步不下降的比例／下降步數）

| 模型與符號 | cone4 | dim_orth_rand4 | dim | 判準 |
|---|---|---|---|---|
| Qwen sell→buy | 0.833／1 | 0.833／1（平手） | 0.667／2 | 不成立（與 rand4 平手） |
| Gemma sell→buy | 1.000／0 | 0.833／1 | 1.000／0 | 成立 |
| Gemma buy→sell | 0.667／2 | 0.833／1 | 0.833／1 | 不成立 |
| GLM buy→sell | 1.000／0 | 0.833／1 | 1.000／0 | 成立（單符號） |
| GPT-OSS sell→buy | 0.833／1 | 0.833／1（平手） | 0.833／1 | 不成立（與 rand4 平手） |
| GPT-OSS buy→sell | 0.667／2 | 0.667／2（平手） | 0.833／1 | 不成立 |

Qwen buy→sell 與 GLM sell→buy 沒有來源類別分母，不評。模型層級（兩個符號都須成立）：Qwen、Gemma、GPT-OSS 不成立；GLM 只評一個符號且成立。

## 解讀

- Gemma sell→buy 與 GLM buy→sell：cone4 與 DIM 完全同樣單調（1.0，0 次下降），優於 `dim_orth_rand4` 只因後者在高劑量崩壞；這不是 cone 的優勢。
- Qwen：cone4（0.833）高於 DIM（0.667），但與 `dim_orth_rand4` 平手，依協議「嚴格高於」不成立。
- 曲線的下降幾乎都發生在最高劑量端（例如 Qwen α=64 的 flip rate 為 0），來自輸出崩壞；這是刻意保留的（未解析算未翻）。
- 這是探索性、事後新增的分析（margin 版結果已知），不改 C8，也不改任何先前版本。

## Artifact

`artifacts/<slug>/concept-cone-steering/runs/flip-monotonicity-v1-20260930-01/result.json`（四個模型各一份，含輸入檔 SHA-256 與各 operator 的曲線）；腳本 `scripts/summarize_flip_monotonicity.py`，測試 `tests/test_flip_monotonicity.py`。
