# Entity Cell: Proposal E4（Residual Stream 壓抑 Readout 比較）

**Document status:** proposed（非 frozen protocol）。本文件不修改任何 frozen
protocol（E1 V3、E2、E3 皆維持原狀），只為 4 個 frozen V3 entity cells 新增一個
表示層診斷實驗。實驗以 proposed probe script 執行
（`entity_cell_readout_delta_probe.py`，見
[research scripts reference](../../research-scripts.md)）。

**Model for run:** Qwen3.5-4B (`.cache/models/qwen3.5-4b`)

**Depends on:**
- Frozen V3 entity cells（calibration + hold-out 雙驗證）：JNJ `(L4, N7676)`、
  PLTR `(L2, N5003)`、BAC `(L0, N7801)`、CAT `(L2, N7997)`，見
  [E1 V3 proposal](proposal-v3.md) 與 [收線報告](../report.md)
- 正式 V3 run 的 fact frames 與人驗 gold：
  `entity-cell-e1-v3-hfm2-calibration-v1`、`entity-cell-e1-v3-holdout-v1`
- Canonical lens：pinned HF pretrained
  （`neuronpedia/jacobian-lens`，revision `a4114d77…`，wikitext calibration，
  sha `c2e20eb4…`），2026-09-07 安裝（見
  [battery 報告 lens blocker 段落](report-jnj-jpm-battery.md)）

---

## 1. 術語定義

**Phase E4（Residual stream 壓抑 readout 比較）**：對單一 frozen entity cell，
在同一 fact frame 的同一 readout 位置，比較「壓抑前（clean）」與「壓抑後
（intervention）」兩條件下 residual stream 的逐層 transported representation
readout，並描述該 cell 對 residual stream 的貢獻在第幾層開始可測、壓抑後如何
逐層消失。這是表示層的診斷實驗：壓抑本身是干預，但 readout 的差異是
**representation-level 觀察**，不建立決策因果、chain-of-thought 或離散
reasoning path 的證據（與 E2 readout 相同的 interpretation limit）。

與既有量的關係：fact gate（V3 Gate 4）量測的是**輸出層**的 gold 序列 joint
log-probability（只有 final layer 一個量）；E4 把同一干預下的量測展開到
**每個 source layer**，回答「cell 的貢獻在哪一層進 residual stream」。

## 2. 動機

1. V3 已確認 4 個 entity cell：壓抑它們造成事實回想崩塌（JNJ F0 −8.51 nats、
   BAC F0 −7.77、PLTR 與 CAT 中等），這是輸出層的因果證據。
2. E2 readout（head-component，`entity-cell-e2-hfm2-readout-v1`）已顯示
   top-ranked attention heads 承載的是 entity 無關的抽象語義簇，沒有 entity
   名稱 token。
3. 未回答的問題：entity cell（MLP 神經元）承載的 entity 專屬事實內容，在
   residual stream 的**哪一層**變得可讀？壓抑前後，逐層 readout 的 full
   vocabulary 分布差異集中在哪些層？這是在輸出層量測無法回答的表示層問題。

## 3. 實驗設計

### 3.1 Targets 與 frames

4 個 frozen entity cells；frames 限於該 entity 在正式 V3 run 中**人驗通過**
的 fact frames（fail-closed）：

| Entity | Cell | Frames | Source run |
|---|---|---|---|
| JNJ | (L4, N7676) | F0, F2, F3 | calibration |
| PLTR | (L2, N5003) | F0, F2, F3 | calibration |
| BAC | (L0, N7801) | F0, F2, F3 | hold-out |
| CAT | (L2, N7997) | F2 只有（F0 為拒答 ` ___?`、F3 年份錯誤，人驗否決） | hold-out |

共 10 個 (entity, frame) pairs。Gold 綁定：probe 重算 clean greedy 前 3
token（確定性），必須與 source run 記錄的 `gold_token_ids` 完全一致，否則
fail-closed 中止。

### 3.2 條件（4 個；劑量與 fact gate 相同：−3.0、all_positions）

| 條件 | 干預 | 期望 |
|---|---|---|
| clean | 無 | baseline |
| target | 該 entity 自己的 cell | readout 差異顯著（entity-specific） |
| matched_random | 同層 matched neuron（`select_matched_random_neuron`，stats 取自 source run 的 `e1/baseline_stats.json`） | 差異 ≤ 0.3 nats（fact gate 控制門檻） |
| wrong_entity | 配對的另一個 frozen cell（JNJ↔PLTR、BAC↔CAT） | 差異 ≤ 0.3 nats |

### 3.3 Readout 定義

- **位置**：fact frame 的最後一個 prompt token（生成第一枚 gold token 的
  位置）；prompt 編碼與 fact gate 相同（raw text、`add_special_tokens=True`、
  無 chat template）。
- **層**：L0–L30 經 canonical lens transport（source_layers 完整覆蓋）；L31
  為 identity readout（lens contract 的 final block J = I），明確標記
  `is_transport=false`。
- **計分**：transported residual 經 FP32 final norm + LM head
  （`fp32_next_token_log_probs`，v2 儀器，與 E2 readout 同一計分路徑），
  取 full vocabulary log-probability。
- **每 (condition, layer) 輸出（compact）**：gold 第一枚 token 的 log-prob、
  top-10 token 與其 log-prob。不持久化任何 raw vector。
- **Delta 量**（相對 clean）：
  - `delta_gold_logp` = suppressed − clean（每層）
  - `first_divergence_layer` = 第一個 |Δ| ≥ 0.5 nats 的層（fact gate 效應
    門檻 `FACT_EFFECT_THRESHOLD`）
  - `max_abs_delta` 及其所在層
  - clean vs target 的 top-10 overlap（|intersection|/10，每層；summary 記
    min/max）
  - 控制組 max |Δ|（matched_random、wrong_entity 各自跨層最大）
- **一致性錨點**：同時以 `score_token_ids` 重測 3-token gold joint
  log-probability（輸出層、與 fact gate 同定義），對照 frozen V3 記錄的
  collapse 值（如 JNJ F0 target −8.51）；不一致超過容差即 fail-closed。

### 3.4 Provenance

輸出 JSON 的 provenance 包含：model、device、dtype、計分路徑（v2 儀器）、
lens path + sha256 + HF revision + repo_id、劑量與門檻、各 target 的 source
run manifest sha、gold 綁定驗證結果、readout 位置定義。

## 4. 解讀限制

1. Readout 是 transported representation readout：token ranks 與 scores 都是
   描述性的，不建立 attention、chain-of-thought、離散 reasoning path 或
   standalone causal 證據（與 E2 readout 的 interpretation_limit 相同）。
2. 壓抑是線性 channel scaling（α = −3.0），readout Δ 是該干預下的
   representation-level 觀察；「first_divergence_layer」描述貢獻何時可測，
   不是精確的寫入層定位（MLP 輸出加入 residual 的層與首次可測層可差 1）。
3. Lens 條件為 pinned HF pretrained（wikitext calibration）；與本地雙語
   lens 的讀數屬不同實驗條件，不可無標示混用（見
   [Qwen Jacobian-lens selection proposal](../../jacobian-lens-selection/proposal.md)）。
4. 計分在 v2 儀器（FP32 final norm 由模型 module 計算）下產出。

## 5. 狀態與 promotion 條件

- **proposed**：以 probe script 執行，產出 compact JSON + 報告
  （`report-e4-readout-delta.md`）。
- **若結果顯示穩定的 entity-specific 逐層模式**（target Δ 在至少一個 frame
  超過效應門檻且控制組 ≤ 控制門檻、first_divergence_layer 在該 entity 的
  多個 frame 間一致、wrong_entity 不重現 target 模式），可討論提升為 frozen
  protocol；依 [experiment versioning](../../documentation-system.md#experiment-versioning)
  另立版本編號與 frozen 文件，不把 proposed 結果回填成 frozen 判定。
