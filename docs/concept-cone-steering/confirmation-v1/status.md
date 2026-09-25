# confirmation-v1：狀態

**狀態：**smoke 進行中（2026-09-25）；尚無 full 結果。協議見 [proposal.md](proposal.md)。

## CPU 前置（commit `e5b4a96`）

- freeze：`artifacts/<slug>/concept-cone-steering/runs/confirmation-v1-freeze-20260925/manifest.json`。四模型五個 family 的 K 等於登錄表（Qwen 100、Gemma 100、GLM 98、GPT-OSS 99），steer-suffix ids 相同；balanced hash 等於 V2 smoke-01／Qwen c2-guided-paper；匿名身分無撞名；GPT-OSS 日期固定成立；Gemma double-BOS。
- A1：`audit-v1-20260925/decision_boundary.{json,md}`。Qwen V1 tokenwise 重現 166 列分歧（146 列為 `{\n` 路徑），其中 165 列 |margin| < 0.5。R0 未完成時跑的 GLM／Gemma 列需在 R0 完成後重跑。
- A2：`audit-v1-20260925/c2_overlap_exclusion.json`。排除 89 家受測重疊公司後，四模型 instruction peak 與 band 都不變。

## Smoke

- Qwen3.5-4B，Colab L4（torch 2.11、transformers 5.14.1；checkpoint 的 config／tokenizer_config／chat template SHA 與本機相同），run `confirmation-v1-colab-smoke-01`，`gates` 以外全部 arm，先 `--max-rows 40` 中斷再續跑：全部完成。結構零層、self-patch（0.0）、self-patch 生成皆為 no-op；dose 表符合構造；teacher-forced steer-suffix 曲線峰值 L15–16（T≈0.37），band L15–17。產物未同步回本機 artifacts（硬體不同，只作 code-path 檢查）。
- Smoke 後修訂（任何 full 結果之前）：single-neuron 目標改為 `mean_p d̂[p]`；R7 summary 加 realized-path margin 位移。見各分項協議的修訂紀錄。
- 本機 Gemma／GLM／GPT-OSS smoke（含 gates）：排程中（GPU1，R0 的 Gemma 完成後）。
