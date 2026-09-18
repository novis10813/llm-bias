# M6-V2：Current-runtime compatibility validation

**版本狀態**：已核准實作；不得回寫 M6-V1 或 Selective-intervention V1 結果。
**研究線**：Selective-intervention M6 external-population validation。
**基礎協議**：[M6-V1 proposal](proposal-m6.md)。

## 1. 為何建立 V2

M6-V1 要求以目前 runtime 重建的 `μ̄` digest 必須與 V1 artifact 完全相同。對既有 V1 artifact 執行 real-model smoke 時，固定 Phase 2A source rows 的目前重建 digest 為
`f1fcf6b4ccbe0b44592b4a624dc0c45e7c79b345df3e23367757d7e9f7a8d729`，而 V1 artifact 記錄為
`99e5dcfb1980b423392175646ffeaeb9a8c9d7844f19aa71c14e3d7071f0aa42`。

因此不能把目前 runtime 的 reconstruction 冒稱為 M6-V1 的 exact frozen-center validation。V2 保留 V1 的 16-company source rows、E-01 `V₈` basis、L15 instruction-span scope、外部 12-company manifest、primary estimand、bootstrap 與 interpretation gate；唯一版本化差異是：V2 使用目前 Hugging Face default runtime deterministic 重建的 `μ̄`，並將 current digest 與 V1 expected digest 同時記錄。

V2 不重新 fit basis、不使用 external rows 建立 center、不調整 alpha、selection、gate 或 external company list。V1 與 V1 的 full-strength negative verdict 不被改寫。

## 2. 固定設定

- model：目前 Hugging Face default cache 的 Qwen3.5-4B，bf16；執行時使用 `CUDA_VISIBLE_DEVICES=1` 將物理 GPU1 映射為 logical `cuda:0`。
- generation：greedy，`do_sample=false`、`temperature=0.0`。
- scoring：fixed-answer `log P("buy") - log P("sell")` margin；margin 不經 generation，也不受 temperature 影響。
- operator：E-01 的既有 `V₈`，不重算、不更新。
- center：16 家 source rows 的 deterministic in-memory reconstruction；不得保存完整 center tensor。
- external population：沿用既定 manifest 的 12 家、四產業各 3 家、48 prompts。
- random control：沿用既定 random 8D basis 與 seed。

## 3. V2 provenance 與解讀

V2 `forward/metadata.json` 與 `analyze/summary.json` 必須記錄：

1. V1 expected L15 center digest；
2. V2 current reconstructed L15 center digest；
3. `center_digest_matches_v1=false`；
4. model/runtime/dtype 與 greedy generation 設定；
5. V2 protocol path。

V2 的 primary result 可回答「目前 runtime 下、沿用 E-01 `V₈` 的 deterministic current-center operator 是否在 external population 顯示 spread reduction」，不能回答「與 V1 artifact 完全相同的 frozen `μ̄` operator 是否泛化」。報告必須保留這個 compatibility boundary。

## 4. 執行順序

1. 在 GPU1 執行 `--smoke`，確認 source reconstruction、V2 digest recording、L15 transform 與 greedy generation。
2. smoke 成功後才執行 formal V2 external run。
3. V2 結果不得回填 M6-V1 report，也不得 rescue 或 overturn V1 negative verdict。
