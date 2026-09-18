# M6：Selective Intervention 子空間的跨公司外部驗證

**版本狀態**：提案草案，尚未凍結，尚未執行。
**研究線**：Selective-intervention M6 external-population validation。
**對象模型**：Qwen3.5-4B（bf16，32 層，hidden size 2560）。
**上游依賴**：`entity-to-dial-e-01` 的 `V₈` basis；Selective-intervention V1 的 `μ̄`、L15、instruction-span scope、projection transform 與原始門檻；Balanced Evidence Gap Phase 2A 的 frozen shared-evidence prompt template。
**本版本不改寫**：Selective-intervention V1 的結果、frozen decision table 或 full-strength 負結果。

## 1. 動機與待測問題

Selective-intervention V1 在原始 16 家公司 population 內，使用 L15 instruction-span 的 k=8 entity-difference subspace 移除，將 group gap 縮減 53.8%，但 full-strength 同時造成 G3/G4 全局副作用。V1 的 `μ̄` 由同一批 16 家 named companies 計算，且 primary group gap 使用的 TOP/BOTTOM 四家公司也參與上游 entity-to-dial 方向建立。

這種設計可以支持原始 population 內的 efficacy，不能單獨支持新公司的泛化。M6 固定使用 V1 已產生的 operator，直接測試未參與 basis 建立、centering 或 V1 formal evaluation 的外部公司：

> **目前凍結的 k=8 subspace removal operator，能否在新公司上縮減 company-level margin spread，且效果仍特異於該 subspace？**

M6 是機制泛化檢驗，不是 V1 的 deployment rescue。M6 任何結果都不改寫 V1 的 full-strength 負結果。

本實驗不是文獻數值復現；它是對本 repo V1 操作的外部 population extension。方法風險包括外部樣本數只有 12 家、IQR 對小樣本排序敏感，以及 fixed-answer margin 與 greedy generation 可能不一致。

## 2. Population 與 frozen operator

### 2.1 Fit/source population：既有 16 家

Fit/source population 指 V1 已使用的 16 家公司。M6 不重新 fit，也不以 M6 的 external companies 更新任何上游產物。

凍結並沿用：

- `entity-to-dial-e-01` 的 `pca_basis_vectors` 前 8 個右奇異向量 `V₈`；
- Selective-intervention V1 定義的 `μ̄`；
- L15 post-block 掛鉤點；
- instruction-span position scope；
- projection removal transform：
  \[
  h'[p] = h[p] - \alpha V_8V_8^T(h[p]-\bar{\mu}[p]);
  \]
- V1 的 model、tokenizer、chat template、prompt family、bf16 dtype 與 answer-token scoring 語義；
- V1 的 random 8D control family、G3/G4 門檻與 α grid。

V1 只保存 `μ̄` 的 digest，不保存可直接重用的完整 center tensor。M6 prepare 可以依 V1 完全相同的 16 家 source rows、reference row、position mapping 與 FP32 計算規則，在記憶體中重建 `μ̄`；這是對 frozen source artifact 的 deterministic reconstruction，不是用 external population 重新 fit。重建後必須與 V1 記錄的 `μ̄` digest 相符，且不得把 center tensor 寫入 artifact。`V₈`、`μ̄` digest、模型 identity、prompt 常數及 V1 transform 版本都必須以 artifact manifest 與 SHA-256 驗證。任一 provenance 不符則 fail-closed，不進入 forward。

### 2.2 External evaluation population：全新 12 家

External population 固定為 12 家新公司。它們不得出現在下列 fit、calibration 或 evaluation 步驟：

1. `entity-to-dial-e-01` 的 basis 建立或 direction source；
2. Selective-intervention V1 的 `μ̄` calibration；
3. Selective-intervention V1 formal evaluation；
4. M6 的 operator construction、selection criteria tuning 或任何結果分析。

它們可以出現在 M6 的 eligible pool，因為 selection 必須從候選池中產生 final 12-company list；eligible-pool 建立只使用執行前固定的結構性條件，不使用模型輸出或研究結果。

每家公司沿用 V1 的 4 個 variants（2 個 reverse × 2 個 order），因此 named external population 為 **12 × 4 = 48 題**。M6 不使用 V1 的 64 題數字作為 external sample size。

### 2.3 External company selection：執行前固定

Selection 使用與原始 16 家相同的四個產業分類：Health Care、Information Technology、Financials、Industrials。每個產業選 3 家，共 12 家。

選取規則如下：

1. 先以既有 canonical company universe 建立候選池；
2. 依 §2.2 排除 16 家 source population 及所有相關上游／V1 使用公司；
3. 只保留能產生完整 V1 四個 prompt variants、通過 ticker/name/span 結構檢查的公司；
4. 每個產業以固定 seed **42** 從 eligible pool 選 3 家；
5. seed、產業分類、eligible pool、exclusion list、final 12-company list 與 selection manifest 必須在任何 model forward 前寫入 prepare artifact。

Seed 42 是本協議常數，不得根據 eligible pool、margin、模型輸出、候選 spread 或任何 pilot 結果更換。不得掃描多個 seed、比較候選名單後回選 seed，也不得在看到結果後替換公司。

若任一產業的 eligible company 少於 3 家，prepare fail-closed；不得由其他產業補位。若 repo 未提供現成且符合本協議的 external manifest，M6 必須新建專屬 manifest；不得把其他研究線的 split manifest 直接當作 M6 manifest。目前已依本節規則建立候選 manifest：`data/baseline/selective-intervention-m6/external-manifest.json`。該檔案只記錄 selection provenance，尚不代表 formal run 已授權。

## 3. Prompt、計分與分析單位

### 3.1 Prompt 與位置

M6 沿用 V1 的 frozen balanced-evidence prompt family：每家公司使用相同的 2 正 2 負 shared-evidence 結構，只改變公司 identity 與 V1 的 reverse/order variant。tokenizer、chat template、instruction span mapping、answer prefix 與 company identity 欄位必須與 V1 一致。

所有 external rows 必須通過：

- 4 個 variants 完整存在；
- instruction span 非空且符合 V1 position contract；
- clean forward 的 output finite；
- prompt metadata 與 external selection manifest 一致。

### 3.2 Primary score

主要 score 沿用 V1 的 fixed answer-token margin：

\[
M(p)=\log p(\text{buy})-\log p(\text{sell})
\]

計分使用 V1 的 FP32 next-token log-prob 語義。M6 不把 margin 位移直接寫成生成行為改變。

為符合 behavior-level reporting contract，M6 另記錄 named external prompts 的 greedy generation：

- clean decision；
- α=1 主臂 decision；
- random-control decision；
- clean→intervention decision-flip rate。

generation flip rate 為描述性結果，不取代 primary spread estimand，也不因 margin 位移而推定 flip。

### 3.3 Company-level aggregation

先對每家公司 4 個 variants 的 margin 取平均：

\[
\bar M_c=\frac{1}{4}\sum_{v=1}^{4}M(c,v)
\]

External clean spread 與 intervention spread 都在 12 個 `\bar M_c` 上計算：

\[
S=Q_{0.75}(\{\bar M_c\})-Q_{0.25}(\{\bar M_c\})
\]

不得對 48 個 raw prompt margins 直接取 IQR。所有 company-level paired statistics 以公司為重抽樣單位。

## 4. Hypotheses、estimands 與 controls

### 4.1 Primary estimand

\[
R_{\mathrm{external}}=
\frac{S_{\mathrm{external},\alpha=1}}
{S_{\mathrm{external},\mathrm{clean}}}
\]

若 clean spread 為 0、非 finite 或不符合 V1 的資料契約，M6 primary gate fail-closed，不能改用其他 spread 定義救援。

### 4.2 Primary hypothesis

**H-M6-1：** 固定的 V1 k=8 operator 在新公司上可使 company-level spread 至少縮減一半：

\[
R_{\mathrm{external}}\leq 0.5
\]

α=1 是 primary strength。V1 的 α ∈ {0.25, 0.50, 0.75, 1.00} grid 可完整重跑作為 dose descriptive curve，但 M6 不依 external 結果調整劑量，也不事後選擇新的 primary α。

### 4.3 Specificity control

M6 在同一 external population、相同 layer、position scope、α 與 centering 下，使用 V1 frozen-seed random orthogonal 8D subspace 作 control。

Define：

\[
D_{\mathrm{main}}=1-R_{\mathrm{external,main}},
\qquad
D_{\mathrm{random}}=1-R_{\mathrm{external,random}}
\]

Random-control specificity criterion 沿用 V1 邏輯：

\[
D_{\mathrm{random}}\leq 0.25D_{\mathrm{main}}
\]

若 `D_main ≤ 0`，specificity criterion 記為 not-evaluable，M6 不得宣稱 subspace-specific efficacy。

### 4.4 Secondary safety diagnostics

M6 重跑 V1 的 external side-effect diagnostics，但不把它們放入 primary mechanism gate：

- **G-M6-3′**：48 個 named external prompts 的 mean margin shift 絕對值 ≤ 0.15 nats；
- **G-M6-4′**：identity-stripped anonymous prompt 的 margin shift 絕對值 ≤ 0.10 nats。

Anonymous prompt 仍使用 V1 的 identity-stripped construction，並在 prepare 階段驗證 prompt string 與 token mapping。M6 不用 external companies 重算 `μ̄`。

## 5. Bootstrap 與 CI

IQR 是 12 家公司上的 order-statistic，M6 不把 nonparametric bootstrap CI 當成小樣本已被解決的證據。CI 只用來標示估計精度並降低把不精確誤判成 null 的風險。

執行前固定：

- paired company-level bootstrap；每次重抽樣同一批 company IDs，再計算 clean 與 α=1 的 spread ratio；
- bootstrap replicate 數：**10,000**；
- CI 算法：**percentile 95% CI**；
- bootstrap seed：**42**；
- 不在看到結果後改用 BCa、改變 replicate 數或改變 seed。

### 5.1 三級 primary interpretation

M6 primary interpretation 同時記錄 point estimate 與 CI：

1. **M6 confirmed**：`R_external ≤ 0.5`，且 95% CI 上界 `≤ 0.5`；
2. **M6 suggestive / unresolved**：point estimate `≤ 0.5`，但 CI 上界 `> 0.5`；
3. **M6 fail**：point estimate `> 0.5`。

`M6 suggestive / unresolved` 不得寫成已確認泛化，也不等於證明沒有外部效果。報告必須列出 12 家公司、IQR、ratio、bootstrap CI，並明確保留 n=12 的限制。

## 6. Frozen decision table

| 結果組合 | M6 解讀 | V1 狀態 |
|---|---|---|
| M6 confirmed + specificity pass | k=8 subspace 的 spread-reduction effect 在新公司上獲得外部確認，且勝過 random 8D control | V1 full-strength 負結果維持不變 |
| M6 suggestive + specificity pass | 外部 spread 有方向一致的縮減訊號，但 12 家樣本不足以確認至少縮減一半 | V1 full-strength 負結果維持不變 |
| point estimate pass + specificity fail | spread 有縮減，但不能歸因於 k=8 subspace；可能是一般 L15 instruction-state 擾動 | V1 full-strength 負結果維持不變 |
| M6 point estimate fail | 未支持 fixed k=8 operator 在 external population 上縮減 spread；不得宣稱 entity-general | V1 full-strength 負結果維持不變 |
| G-M6-3′ 或 G-M6-4′ fail | external population 也出現 side effect；記為 safety diagnostic，不改寫 primary mechanism conclusion | 不救援，也不加重 V1 判定 |
| primary pass 但 generation flip 為 0 | margin spread 受到影響，但沒有證據顯示 greedy generation decision 改變 | 不得把 margin effect 寫成 generated decision effect |

M6 的「外部確認」只表示 frozen operator 對新公司的 spread estimand 泛化。它不表示 operator 可部署、不表示 full-strength removal 沒有副作用，也不表示已找到唯一或必要的 entity circuit。

## 7. Boundary、fail-closed 與停止條件

1. **不重算上游產物**：不得用 external company 更新 `V₈`、`μ̄`、layer、position scope、centering、α primary 或 random basis。
2. **不挑公司**：不得根據 clean margin、α=1 margin、spread、decision flip 或任何模型輸出替換 12 家名單。
3. **不調劑量**：α=1 primary gate 固定；dose curve 只作 descriptive。
4. **不使用 TOP/BOTTOM group gap**：external 12 家沒有預先凍結的 TOP/BOTTOM，因此 M6 不計算或補造 group gap。
5. **不把 G3′/G4′ 當作機制 gate**：它們描述 external side-effect profile，不改寫 V1 full-strength verdict。
6. **資料不足即停止**：任一產業少於 3 家 eligible company、任一公司缺少完整 4 variants、V1 provenance 不符、anonymous mapping 不一致、clean margin 非 finite 或 clean spread 為 0，prepare fail-closed。
7. **CI 不足不補樣**：若 bootstrap CI 過寬，記為 `suggestive / unresolved`；不得臨時增加、替換或挑選公司來改善精度。增加樣本須另立版本。
8. **版本分立**：改變 external selection rule、12 家名單、primary estimand、IQR aggregation、bootstrap method、CI interpretation、random control、layer、position scope、centering、α primary 或 decision threshold，建立 M6 新版本，不回填本協議。

## 8. Input、output 與 provenance

### 8.1 Inputs

Prepare 必須記錄並驗證：

- `entity-to-dial-e-01` manifest、summary SHA-256 與 `V₈` provenance；
- Balanced Evidence Gap Phase 2A manifest、canonical four-variant prompt template 與 source hash；
- Selective-intervention V1 manifest、V1 `μ̄` digest、transform constants 與 prompt contract；
- external selection manifest、eligible pool、exclusion list、final 12-company list、sector labels、selection seed；
- model、tokenizer、chat template、dtype 與 revision identity；
- prompt rows、variant labels、instruction span mapping；
- frozen random-control seed。

### 8.2 Compact outputs

M6 只輸出 derived compact data：

- per-company、per-variant clean／α arm／random margin 與 decision；
- per-company mean `M̄_c`；
- clean、main、random 的 spread、spread ratio、reduction ratio；
- paired company-bootstrap CI、replicate count、seed、method；
- G-M6-1、G-M6-2、G-M6-3′、G-M6-4′ 結果；
- generation flip counts/rates 與 control flip counts/rates；
- manifest、input hashes、selection provenance 與 finite-value checks。

不得保存 raw activations、residuals、hidden states、gradients、Jacobian、KV cache 或 `μ̄`／`V₈` 的未經必要 artifact contract 管理之 raw tensor。M6 只消費既有 frozen basis 與 center；formal artifact 只保存其 digest 與 compact derived statistics。

## 9. Planned execution contract

本協議目前沒有已實作的 M6 operator，formal run 尚未授權。實作完成後，operator 應提供 shared workflow 的 `prepare → forward → analyze → finalize` lifecycle，並在 formal run 前通過真實模型與 tokenizer smoke，涵蓋：

- external selection manifest 與 exclusion checks；
- frozen `V₈`／`μ̄` provenance 驗證；
- clean、main、random、anonymous no-op／finite checks；
- company-level aggregation 與 paired bootstrap；
- generation secondary metric；
- compact artifact schema 與 manifest hash。

規劃中的 CLI path 為：

```bash
uv run python scripts/selective_intervention_m6.py \
  --model .cache/models/qwen3.5-4b \
  --source-basis-run artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-e-01 \
  --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \
  --v1-run artifacts/qwen3.5-4b/selective-intervention/runs/selective-intervention-v1-gpu-bf16-01 \
  --external-manifest <frozen-m6-external-manifest> \
  --run-id <m6-run-id>
```

此命令是實作契約草案；在 operator、schema、smoke 與 provenance checks 完成前，不得用它宣告 formal run 可執行或授權 real-model run。

## 10. 查證與版本界線

- 上游 basis 與 state-difference 結論：[Entity-to-Dial Phase E](../../entity-to-dial/details/proposal-phase-e.md)。
- V1 transform、population、G1–G4 與 full-strength verdict：[Selective-intervention V1 proposal](proposal-v1.md)、[V1 report](../report.md)。
- M6 不回寫 V1 原始結果；M6 result 應另立報告並直接引用本協議。
- M6 的 external manifest 應在 prepare artifact 中保存 provenance；若後續改名單或選取規則，建立 M6 新版本。

## 11. Revision history

| Rev | 日期 | 狀態 | 說明 |
|---|---|---|---|
| 0 | 2026-09-18 | proposed | 建立 M6 外部 population protocol：凍結既有 16 家產生的 `V₈` 與 `μ̄`，以四產業分層、每產業 3 家選取 12 家新公司；primary estimand 改用 company-level IQR spread；加入 paired company-bootstrap CI、三段式 precision interpretation、random specificity control、G3′/G4′ secondary diagnostics 與 V1 non-revision boundary。 |
