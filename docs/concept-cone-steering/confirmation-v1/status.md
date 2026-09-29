# confirmation-v1：狀態

**狀態：**四模型 full 全部完成（2026-09-26，run `confirmation-v1-20260925-full-01`：Qwen3.5-4B、GLM-4-9B、Gemma-4-12B、GPT-OSS-20B）。協議見 [proposal.md](proposal.md)。

## CPU 前置（commit `e5b4a96`）

- freeze：`artifacts/<slug>/concept-cone-steering/runs/confirmation-v1-freeze-20260925/manifest.json`。四模型五個 family 的 K 等於登錄表（Qwen 100、Gemma 100、GLM 98、GPT-OSS 99），steer-suffix ids 相同；balanced hash 等於 V2 smoke-01／Qwen c2-guided-paper；匿名身分無撞名；GPT-OSS 日期固定成立；Gemma double-BOS。
- A1：`audit-v1-20260925/decision_boundary.{json,md}`。Qwen V1 tokenwise 重現 166 列分歧（146 列為 `{\n` 路徑），其中 165 列 |margin| < 0.5。R0 未完成時跑的 GLM／Gemma 列需在 R0 完成後重跑。
- A2：`audit-v1-20260925/c2_overlap_exclusion.json`。排除 89 家受測重疊公司後，四模型 instruction peak 與 band 都不變。

## Smoke

- Qwen3.5-4B，Colab L4（torch 2.11、transformers 5.14.1；checkpoint 的 config／tokenizer_config／chat template SHA 與本機相同），run `confirmation-v1-colab-smoke-01`，`gates` 以外全部 arm，先 `--max-rows 40` 中斷再續跑：全部完成。結構零層、self-patch（0.0）、self-patch 生成皆為 no-op；dose 表符合構造；teacher-forced steer-suffix 曲線峰值 L15–16（T≈0.37），band L15–17。產物未同步回本機 artifacts（硬體不同，只作 code-path 檢查）。
- Smoke 後修訂（任何 full 結果之前）：single-neuron 目標改為 `mean_p d̂[p]`；R7 summary 加 realized-path margin 位移。見各分項協議的修訂紀錄。
- 本機 Gemma／GLM／GPT-OSS smoke（含 gates）：排程中（GPU1，R0 的 Gemma 完成後）。

## Full runs（lab jobs，commit `06efa5d`，host idlab）

- `confirm-4`（Qwen3.5-4B，`--arms all`，先 `--max-rows 40` 再 full）與 `confirm-6`（GLM-4-9B，tier1→tier2）於 2026-09-26 09:37／09:42 UTC 完成全部 arm；`confirm-5`（GLM 首次嘗試）在 setup 的 `cp third_party/jacobian-lens` 失敗後以 `confirm-6` 取代（setup 已含 `mkdir -p third_party`）。
- 09-26 後 tmux session 消失、本機 `lab.db` 停在 running；2026-09-29 `lab dispatch` reconcile 後兩 job 皆記為 succeeded（exit 0）。兩支 log 的最後一行為一個多餘 process 的 CUDA-init warning（無 invocation 記錄、無輸出、無 result 改寫，對結果無影響）；host 上查無 cron／systemd watcher。
- 完成度核验：Qwen 17/17 arm、GLM 15/15 arm（`TIERS` 定義，dim_layers／shuffle 不適用 GLM）的 `<arm>/result.json` 皆 `complete=true`；`job_summary.json`（Qwen 為 17-arm 全量 summary；GLM 為最後一次 invocation `ops` 的 summary，各 arm 以 `<arm>/result.json` 為準）與 `invocations.jsonl` 齊備。
- 產物已落 host store：idlab `~/Projects/lab-assets/llm-bias-confirm/artifacts/<slug>/concept-cone-steering/runs/confirmation-v1-20260925-full-01/`（含 `job_output.log` 副本）；`~/.lab/jobs/confirm-{4,6}/output.log` 為原始 log。

## Full runs（續）：Gemma-4-12B、GPT-OSS-20B（idlab2，非 lab）

- 兩模型的 full 在 idlab2 的 `~/llm-bias` clone 手動執行（非 lab job），2026-09-26 完成：Gemma tier1（14 arm）＋ tier2（`shuffle`、`loso_construction`）共 16/16；GPT-OSS tier1（13 arm）＋ tier2（`ops`、`loso_construction`）共 15/15。`<arm>/result.json` 皆 `complete=true`，`job_summary.json` 與 `invocations.jsonl` 齊備；smoke-01 與 freeze 同在。
- 源位置：idlab2 `/mnt/train-data-1-hdd/sam/llm-bias/artifacts/<slug>/concept-cone-steering/runs/confirmation-v1-20260925-{full-01,smoke-01}` 與 `confirmation-v1-freeze-20260925/`；已同步至 idlab lab-assets 與本地 `artifacts/`（本地同步記錄見下）。

## 本地同步（2026-09-29）

- `artifacts/` 與 `data/` 已自 idlab2（舊 clone，不含 `artifacts/archive`）與 idlab（lab-assets ＋ confirm-3 worktree 的 Qwen smoke）rsync 回本地；模型 checkpoint 不同步（留在遠端）。
