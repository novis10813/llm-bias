# Investment-dial

本目錄保存 investment-dial 實驗線的 canonical 文件。

**線狀態**：收線（2026-09-09）。線內問題已由三次 run 完整回答；收線總結見
[結案報告](report-line-closing.md)。

## Paper provenance

本線是以下論文的 investment-bias dial 方法復現：

- Park, S., Park, S., Lee, H., Kwon, G., Ahn, W., Choi, J., Lopez-Lira, A., Kim, Y.,
  Choi, C., Kong, H., and Lee, Y. (2026). *Your AI, On a Dial: Controlling Investment
  Bias in LLMs with a Single Neuron*. arXiv:2608.22852（preprint, under review）。

論文提出在 inference time 對單一 MLP down-projection coordinate 施加強度 Δ 的
additive intervention，連續校準 model 整體 buy/sell stance；coordinate 經
decision-relevance screening、A/B split 的 transfer curve calibration 與 RMSE ranking
選出；response 以 balanced bullish/bearish evidence protocol 下的
π = (buy − sell) / (buy + sell) 量測（只有 buy/sell、JSON decision 與 reason、
decision option 順序 counterbalanced、256-token generation budget）。

本機 implementation（`llm_bias/investment_dial`，protocol version
`investment-dial-local-v1`）遵循論文的 selection 與 calibration 協議：
screen → A/B calibration → RMSE ranking → 對 target grid {−0.3, 0, +0.3} 的
coefficient fitting。已知差異：

- 論文評估 5 個 open-weight LLM（含 Qwen3-8B，selected coordinate L23/F4099，
  RMSE 0.038）；本機 V1 run 使用 Qwen3.5-4B，不是論文的評估模型之一，selected
  coordinate 依論文設計是 model-specific（本機為 L15/n8490）。
- 本機 V1 calibration 是 exploratory run，不是 certified 的復現結果；其 RMSE
  （0.5757）與 curve shape 不與論文的 per-model 數字直接比較。

## 版本矩陣

| 版本 | 文件 | Status | Primary outcome / 證據 |
|---|---|---|---|
| V1 calibration（protocol `investment-dial-local-v1`） | 無 proposal 文件；operation contract 以 run protocol artifact 維護 | implemented；exploratory run 完成 | selected coordinate L15/n8490；B RMSE 0.5757（neutral 與 +0.3 target 各偏 ~0.7） |
| fine-a diagnostic（protocol `investment-dial-fine-a-v1`） | [A-only diagnostic](diagnostic-fine-a.md) / [Fine-a report](report-fine-a.md) | implemented；diagnostic run 完成 | run `fine-a-gpu-bf16-01`：zero 附近有 steep transition，支持（但不單獨證明）coarse-interpolation explanation |
| V2 calibration（protocol `investment-dial-calibration-v2`） | [proposal-v2.md](proposal-v2.md) / [report-v2.md](report-v2.md) | formal run 完成；gate `pass` | run `calib-v2-gpu-bf16-01`：15 點 A curve + fine-grid inversion；B reevaluation RMSE 0.0579、max error 0.0824（gate ≤ 0.15 / ≤ 0.25）、`certified=true`；V1 粗 grid 失敗被修正（支持 coarse-interpolation explanation） |

### 文件索引

- [結案報告](report-line-closing.md)：線收線總結（核心結論、claim 邊界、產物索引、延伸新線候選）。
- [V2 proposal](proposal-v2.md)：fine-grid 重校準與 B reevaluation 的 frozen 契約
  （含 V1 stored summary 的數據品質註記與 recompute rule）。
- [V2 report](report-v2.md)：第一次 formal run（`calib-v2-gpu-bf16-01`）結果與解釋邊界。
- Run artifacts：
  - V1 calibration（exploratory）：
    `artifacts/qwen3.5-4b/investment-dial-calibration/runs/exploratory-v1-20260907-07-gpu0-calibration`
  - Fine-a diagnostic：
    `artifacts/qwen3.5-4b/investment-dial-fine-a/runs/fine-a-gpu-bf16-01`
  - V2 calibration：
    `artifacts/qwen3.5-4b/investment-dial-calibration-v2/runs/calib-v2-gpu-bf16-01`

注意：V1 沒有獨立的 proposal 文件（當時直接以 run protocol artifact 凍結
契約）；V2 起依 [docs/AGENTS.md](../AGENTS.md) 的 versioning 規則以 versioned
proposal 維護。
