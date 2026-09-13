# Entity-to-Dial Phase E：Dual-Block 鑑別與 State-Direction Patch 協議

**狀態**：completed（Rev 1.3；formal run `entity-to-dial-e-01` 完成，2026-09-12；**Gate E1 pass、Gate E2b fail**）  
**對象模型**：Qwen3.5-4B（bf16，hybrid Gated DeltaNet 架構，32 層，full-attention 層為 L3/7/11/15/19/23/27/31；`hidden_size = 2560`、`intermediate_size = 9216`）  
**研究線定位**：承接 Phase D null（instruction span 單一 block additive 貢獻全 null；2B full residual T = 0.47–0.60 in L12–15）的結論，以兩個鑑別實驗確認：（E1）signal 在 joint state 而非任一 block 的 additive 貢獻；（E2）該 joint state difference 是否集中在少數 PCA 方向，以及 dial channel（L15/N8490）在其原生 MLP 空間內是否是 carrier。

---

## 1. 背景：已知曲線與開放問題

### 1.1 2B instruction-span full-residual sweep（完整曲線）

Phase 2B 的 instruction span full-residual swap 覆蓋 L0–31（8 directions），完整 normalized transfer T：

| 層範圍 | T |
|---|---|
| L0–11 | ≈ 0（−0.07 ～ +0.04） |
| L12 | +0.252 |
| L13 | +0.470 |
| L14 | +0.491 |
| L15 | +0.604（峰值） |
| L16 | +0.548 |
| L17 | +0.402 |
| L18 | +0.361 |
| L19–26 | +0.04 ～ +0.11 |
| L27–31 | ≈ 0 |

層局部增量 ΔT（full(L) − full(L−1)）：L12 +0.21、L13 +0.22、L14 +0.02、L15 +0.11；L16 起轉負（−0.06 / −0.15 / −0.04）。**L12–13 是 stance 寫入的主要層，L14–18 是 carryover 與衰減。**

### 1.2 Phase D1 block patch（instruction span，L12–31）

D1 在 L12–31 的 instruction span 上分別替換 MLP block 貢獻與 attention block 貢獻，結果全 null（L12–31 全域 |mean ΔM| 最大 0.091 nats（L12 attn），CI 全跨 0）。Per-direction 符號方向朝 sell（erasure 簽章），與 full residual swap 的 toward-source 正號相反。

### 1.3 未解問題

Full residual swap 有效（T ≈ 0.5），single block patch null 且符號相反。Transfer 效應來自**兩個 block delta 加上 pre-L state difference 的組合**，在下游以非加性方式產生。需要鑑別的假設：

- **H_carryover**（下游近似線性）：joint patch 的效應 ≈ incremental ΔT(L) / T(L)，不自足。
- **H_joint**（block delta 組合自足）：joint patch 的效應 ≈ full(L)，pre-L state difference 冗餘。

對立預測：

| 假設 | joint T(L15) 預測 | joint T(L12) 預測 |
|---|---|---|
| H_carryover（≈ incremental） | ≈ 0.11 | ≈ 0.21 |
| H_joint（≈ full） | ≈ 0.60 | ≈ 0.25 |

L15 差距最大（0.11 vs. 0.60），是鑑別力最強的層。

---

## 2. 術語定義

沿用 [proposal.md §2](proposal.md#2-定義) 和 [proposal-phase-d.md §2](proposal-phase-d.md#2-新增術語定義) 的術語。新增：

- **joint state**：在層 L 的 instruction positions 上，同時替換 attention block delta（`mid − pre`）與 MLP block delta（`post − mid`）後的殘差流狀態。等價算術：`patched_post = post_L_tgt + Δs`，其中 `Δs = (post_L_src − pre_L_src) − (post_L_tgt − pre_L_tgt)`（FP32）。等同於 `post_L_src − (pre_L_src − pre_L_tgt)`。與 2B full residual swap（直接替換為 `post_L_src`）之差正好是 `(pre_L_src − pre_L_tgt)` 項；H_carryover 主張該差異項承載效應，H_joint 主張它冗餘。Joint 只需 `pre` 與 `post`，`mid` 由 `record_block_states` 一併捕獲（免費）但計算上不使用。

- **full-swap transform**（E1 reference arm）：直接以 `post_L_src` 替換 `post_L_tgt[:, instruction_positions, :]`（與 2B 的 `make_span_transform` 算術 bit-exact 相同）；作為 joint ratio 的 in-run 分母，不引用 2B 存檔值。

- **joint ratio**：`joint_T(L) / full_T(L)`，其中分子與分母均為當次 run 的 in-run 量測。Per-direction 計算後取**有效方向子集的中位數**（見 R1 denominator rule）。

- **有效方向**（R1 denominator rule，pre-registered）：`|ΔM_full| ≥ 0.2 nats` 的方向納入 ratio 計算；其餘方向在 artifact 中標注 `excluded: true`，不進入中位數。此門檻 frozen，不事後修改。預期 7/8 方向納入（2B 存檔有一個方向 full T = −0.258，對應 |ΔM_full| 視 group gap 估算可能低於 0.2）。

- **state difference vector**（`Δs_L`）：`source_post_L[:, instruction_positions, :] − target_post_L[:, instruction_positions, :]`，shape `[P, 2560]`，FP32（P = instruction_positions 數，所有 16 家公司的 instruction span 等長且 offset-parallel，P 為常數）。

- **PCA 基底**：對 8 個 direction 的 `Δs_L15` 拼成矩陣 `Z ∈ R^{8P × 2560}`，做 thin SVD，取前 16 個**右奇異向量**（`V ∈ R^{2560 × 16}`，state space 中的方向）作為 state-direction 基底。左奇異向量在 `R^{8P}`（direction×position 空間），不是 state 空間，不使用。

- **projected patch**：只替換 `Δs_L15` 在前 k 個右奇異向量的投影分量：`Δs^{(k)} = V_k V_k^T Δs_L15`，patch 為 `post_L15_tgt[:, P, :] + Δs^{(k)}`（cast 回 bf16）。

- **dial channel transplant**（E2b，pre-registered）：Dial（L15/N8490）是 L15 MLP down-projection 的**輸入 activation** 的 channel 8490（`intermediate_size = 9216`），不在 2560 維 residual 空間。E2b 直接在原生 MLP 空間操作：capture source/target 在 instruction positions 的 `a_{8490}(p)`（使用 `dial_probe` 的 `dense_down_projection` pre-hook），計算 per-position delta `δ(p) = a_src(p) − a_tgt(p)`，patch 時對 target 在 instruction positions 的 channel 8490 加 δ（其他 positions = 0，其他 channels 不動）。這比 Phase C 的 all-positions 統一 δ 更嚴格（position-restricted + 移植真實值差），因此 E2b fail 是 dial 路徑的 definitive null（不是 3A 的重跑）。

- **dial residual footprint**（descriptive arm，不 gate）：`W_down[8490, :]`（L15 MLP down-projection 權重第 8490 行，shape `[2560]`，從模型參數讀取），歸一化為單位向量後投影 `Δs_L15`，patch instruction positions（8 forwards）。此 arm 有 confound（其他 neuron 可能寫同一 residual 方向），僅作描述性副臂。

---

## 3. 假說（pre-registered，frozen 後不調整）

- **H_E1（joint 自足）**：L15 的 joint ratio 的有效方向中位數 ≥ 0.5。等效於 joint T(L15) ≥ 0.30（H_joint 一方）。Gate E1 pre-registered。
- **H_E2a（低維集中）**：L15 的 `effect_ratio(k=8) ≥ 0.8`（PCA 前 8 方向解釋 ≥ 80% 的 full-swap 效應）。Descriptive；不 gate；為後續 Phase F 提供 target dimensionality。
- **H_E2b（dial carrier）**：L15 的 dial channel transplant 的 `dial_ratio`（有效方向中位數）≥ 0.5。Gate E2b pre-registered。Falsifier：若 dial_ratio < 0.05，dial 路徑為 definitive null。

注意：H_E2b 與 Phase C（C2 ratio = 0.051，readout explanation）不矛盾——E2b 量的是「instruction position 上 source/target dial channel 的真實值差，作為 position-restricted transplant，能否恢復 margin」（intervention measure）。兩者可分別成立或失敗。

---

## 4. 實驗設計

### 4.1 公司與 prompt 族

完全繼承 Phase A/B/C/D 的 16 家公司、8 directions（NSC↔IT、NSC↔BDX、BLK↔IT、BLK↔BDX 雙向）、frozen shared-evidence template、canonical variant（reverse=False, order=0）。

**Instruction span 對齊**：16 家公司使用相同 frozen template，tokenized 後 instruction span 等長（P tokens）且 offset-parallel（Phase D Rev 1 定義），patch 為直接索引對應（identity mapping in token space）。

**Smoke pre-check**：group gap ≥ 0.5 nats（從 2A 存檔重算）；self-source no-op |ΔM| ≤ 1e-12。

**Smoke grid**（mechanism validation，不評 gate）：layers {12, 15}；directions {NSC→IT, IT→NSC}；E2 k ∈ {1, 8, full} + dial 臂。驗收條件（全部必過）：
1. 所有 joint no-op |ΔM| = 0.0（bit-exact）。
2. Full 臂在 L12/L15 的 T 落在 2B 存檔值 ±0.05 nats 帶內（L12 ≈ 0.252、L15 ≈ 0.604）。
3. Dial hook 在 instruction positions 正確 capture `a_{8490}`，非 instruction positions 的 channel 8490 不變（position-restriction 驗證）。
4. k=full arm ΔM == full-swap arm ΔM（bit-exact，驗證投影完整性）。
5. PCA 決定性：相同輸入兩次 SVD 輸出 `V` bit-exact 相同。

### 4.2 Sub-experiment E1：Dual-Block Joint Patch + Full-Swap Reference

**問題**：L12–18 的 joint patch ratio 分佈是否支持 H_joint（ratio ≥ 0.5 at L15）還是 H_carryover（ratio ≈ incremental）？

**兩個 arm（per layer per direction）**：

- *joint arm*：patch `post_L_tgt[:, P, :] += Δs_L`（FP32 Δs，cast 回 bf16）。
- *full arm*（reference，in-run）：patch `post_L_tgt[:, P, :] = post_L_src[:, P, :]`（exact swap，bit-exact 與 2B 算術相同）。

**State capture**（per direction）：

- Source：`record_block_states(source, L12–18)` → `{pre, mid, post}` per layer。（Joint 使用 `pre` 和 `post`；`mid` 免費捕獲但不進計算。）
- Target：`record_block_states(target, L12–18)` + FP32 tail margin（live target margin）。

不引用 2B 存檔的 source state：cross-pipeline state 語義即使確認一致，`Δs` 計算需要 FP32，重新 capture 消除 dtype round-trip 疑慮，且成本可接受。

**Forward 協議（per direction）**：

1. Source capture：1 forward。
2. Target capture + live margin：1 forward。
3. Joint patch：7 layers × 1 = 7 forwards。
4. Full-swap：7 layers × 1 = 7 forwards。
5. Joint no-op（`Δs = 0`）：7 forwards（fail-closed）。
6. Full no-op（self-source full swap）：7 forwards（fail-closed，預期 ΔM = 0.0 bit-exact）。

共 **8 × (2 + 7 + 7 + 7 + 7) = 240 forwards**。

**Gate E1**（pre-registered）：L15 的 joint ratio 的有效方向（|ΔM_full| ≥ 0.2）中位數 ≥ 0.5。全部 7 層的 joint ratio 與 incremental ΔT/T 一併輸出（descriptive carryover curve）。

### 4.3 Sub-experiment E2：State-Direction Patch（PCA sweep + Dial transplant）

**問題**：L15 的 `Δs` 是否集中在少數 PCA 方向（H_E2a）？Dial channel 在 MLP 空間的 position-restricted transplant 是否有效（H_E2b）？

**Step E2-0（state capture，共用 E1）**：從 E1 的 capture forward 直接取 8 個 direction 的 `post_L15`（source + target），計算 `Δs_L15`（`[P, 2560]`，FP32）；同時用 `dial_probe` 的 pre-hook 讀取 source/target 的 `a_{8490}(p)` for p ∈ instruction positions（fold 進 E1 capture forward，0 額外 pass）。

**Step E2-1（PCA）**：拼成 `Z ∈ R^{8P × 2560}`，thin SVD，取前 16 個右奇異向量 `v_1, …, v_{16} ∈ R^{2560}`。

**Step E2-2（PCA sweep）**：k ∈ {1, 3, 8, 16, full}，per direction：計算 `Δs^{(k)}`，patch `post_L15_tgt[:, P, :]`，量 toward-source ΔM。`effect_ratio(k)` = 有效方向中位數（`ΔM_k / ΔM_full`）。k=full arm 的 ΔM 即為 E1 L15 full-swap 的 in-run 值（bit-exact，作為 2B L15 存檔的交叉驗證）。

**Step E2-3（Dial channel transplant）**：per direction，在 L15 MLP down-projection 輸入的 channel 8490，instruction positions p 各加 `δ(p) = a_src(p) − a_tgt(p)`（`mlp_addition`-style hook，position-restricted）；非 instruction positions 的 channel 8490 不動；量 toward-source ΔM。`dial_ratio` = 有效方向中位數（`ΔM_dial / ΔM_full`）。

**Step E2-4（Dial residual footprint，descriptive）**：讀取 `W_down[8490, :]`（L15，shape `[2560]`），歸一化；投影 `Δs_L15` 到此方向，patch instruction positions；量 ΔM（8 forwards）。標注 confound（其他 channel 可能共用此 residual 方向）；不進 gate。

**Forward 協議（E2，共用 E1 state）**：

- E2-2（PCA sweep）：8 directions × 5 k-values = 40 forwards；8 no-op（k=full arm）= 8 forwards。
- E2-3（Dial transplant）：8 forwards；8 no-op = 8 forwards。
- E2-4（Footprint，descriptive）：8 forwards。

合計 **72 forwards**（不含 E1 state capture）。

**Gate E2b**（pre-registered）：`dial_ratio`（有效方向中位數）≥ 0.5。  
**E2a**（descriptive）：`effect_ratio(k=8) ≥ 0.8` 作為 target，輸出完整 k vs. ratio 曲線。

---

## 5. 實現

**Package**：`llm_bias/entity_to_dial/`，新增子模組 `joint_patch.py`：

- `make_joint_transform(source_pre, source_post, target_pre, positions)`：計算 `Δs = (source_post − source_pre) − (target_post − target_pre)`（FP32），生成 `ResidualTransform`（`residual_interventions` 在 post-block 點介入；FP32 arithmetic → cast bf16）。算術等價：`make_joint_transform` 的效果 = `make_block_transform(mlp) + make_block_transform(attn)`；unit test 機器驗證此等價性。
- `make_full_transform(source_post, positions)`：exact swap，bit-exact 與 `make_position_transform` 等價；作為 E1 reference arm。
- `pca_state_directions(delta_vectors, k=16)`：`torch.linalg.svd(Z, full_matrices=False)`，返回前 k 個**右奇異向量**（`Vh[:k].T`，shape `[2560, k]`）與 singular values。
- `project_delta(delta, basis_k)`：`(delta @ basis_k) @ basis_k.T`（投影到前 k 向量的子空間）。
- `dial_footprint_direction(model, layer=15, channel=8490)`：從 `dense_down_projection(model.layers[layer])` 讀取 `weight[channel, :]`（shape `[2560]`），歸一化返回。

**Operator**：`scripts/entity_to_dial_phase_e.py`（串行：E1 capture → E1 joint/full patch → E2 PCA → E2 patch/transplant/footprint → analyze）。

CLI 契約：

```bash
# formal run
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/entity_to_dial_phase_e.py \
    --model .cache/models/qwen3.5-4b \
    --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \
    --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \
    --run-id entity-to-dial-e-01

# smoke
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/entity_to_dial_phase_e.py \
    --model .cache/models/qwen3.5-4b \
    --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \
    --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \
    --smoke
```

**Run root**：`artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-e-<seq>/`。

**Stages**：`prepare` → `forward_e1` → `forward_e2` → `analyze` → finalize。

**Artifact schema（compact）**：

- `prepare/rows.jsonl`：16 行（沿用 Phase D schema，加 `instruction_last_position: int`、`instruction_span: [int, int]`）；`prepare/provenance.json`。
- `forward_e1/records.jsonl`：`{phase: "e1", direction, layer, arm ∈ {joint, full, joint_noop, full_noop}, patched_margin, toward_source_delta_m, normalized_transfer, full_t_ref (full arm only), excluded: bool (R1 rule), m_source, m_target}`。
- `forward_e2/records.jsonl`：`{phase: "e2", direction, arm ∈ {pca_k1, pca_k3, pca_k8, pca_k16, pca_full, dial_transplant, footprint, dial_noop, pca_full_noop}, patched_margin, toward_source_delta_m, effect_ratio (nullable), excluded: bool, dial_delta_per_position: [float] (dial transplant arm only; per instruction position Δa_{8490}, descriptive)}`。
- `analyze/summary.json`：`gate_e1`（L15 joint ratio per-direction、有效方向中位數、pass）；`e1_curves`（per-layer {joint_ratio, full_t, joint_t, incremental_t_ref, excluded_directions}）；`gate_e2b`（dial_ratio、有效方向中位數、pass）；`gate_e`（E1 AND E2b）；`e2a_curve`（k vs. effect_ratio，descriptive）；`e2_footprint`（footprint ΔM、descriptive）；`pca_singular_values: [float×16]`；`pca_basis_vectors: [[float×2560]×16]`（top-16 右奇異向量，~400 KB JSON；derived statistic，符合 compact 政策）；`smoke: bool`。
- `manifest.json`（SHA-256 + record counts + stage lifecycle）。

**持久化**：不存 raw activations、`Δs` 向量。持久化 top-16 PCA 右奇異向量（Phase F 啟動所需；derived from direction deltas，不是 raw hidden states）。

**Tests**（`tests/test_entity_to_dial_joint.py`，fake model + mocked forward）：

- `make_joint_transform`：self-source（`source = target`）→ `Δs = 0` → `ΔM = 0.0` bit-exact；已知 `Δs` 的 expected patch output；FP32 cast-back 精度。
- **機器交叉驗證**：`make_joint_transform(pre_s, post_s, pre_t, positions).apply(x)` == `make_block_transform(mlp, ...).apply(make_block_transform(attn, ...).apply(x))`（fake state，驗證 joint = two-block sum）。
- `make_full_transform`：bit-exact 等價於 `make_position_transform`（same fake input）。
- `pca_state_directions`：正交歸一（`V^T V ≈ I`，atol=1e-5）；shape `[2560, k]`；決定性（相同輸入兩次輸出 bit-exact）。
- `project_delta`：k=full → 等同 unprojected（fake random delta in span of basis）；k=0 → `Δ = 0`。
- **Dial hook position-restriction**：instruction positions 的 `a_{8490}` 被正確 capture；非 instruction positions 的 channel 8490 在 transplant forward 後不變（monkeypatched mlp hook）。
- **R1 denominator rule**：`|ΔM_full| < 0.2` 的方向 `excluded=True`，不進中位數；邊界 `|ΔM_full| = 0.2` 的行為（≥ 納入）。
- Pipeline smoke path：E1 + E2 串行，manifest complete，gate 邊界（joint ratio = 0.5、dial ratio = 0.5）。

---

## 6. 成本估算

| Sub-experiment | Forwards | 說明 | 估算 |
|---|---|---|---|
| E1（joint + full）| 240 fwd | 8 × (2 capture + 7 joint + 7 full + 7 joint-noop + 7 full-noop) | ~10 min |
| E2（PCA + dial）| 72 fwd | 共用 E1 state；40 pca + 8 noop + 8 dial + 8 noop + 8 footprint | ~3 min |
| **Phase E 合計** | **~312 fwd** | 單次模型載入，串行 E1→E2 | **~15 min** |

---

## 7. 讀法與後續決策

| Gate E1 | Gate E2b | E2a curve | 含義 | 後續 |
|---|---|---|---|---|
| Pass（ratio ≥ 0.5） | Pass（dial ≥ 0.5） | 低維（k=8 ≥ 0.8） | Stance shift 集中在含 dial channel 的低維子空間；dial 是 carrier | Phase F：沿 dial + top PCA 方向做 fine-grained intervention（精確確認 carrier 組合），使用持久化 basis |
| Pass | Fail（dial < 0.05） | 低維 | Stance shift 低維，但 dial channel 不是 carrier；bias 走 dial 以外的特定方向 | Phase F：以 top PCA 方向做 single-direction `mlp_addition`-style validation |
| Pass | Fail | 高維（k=8 < 0.5） | Joint state 是載體，但 state shift 高維分散；無法用少數方向描述 | 換範式：linear probe 或 population-level representation search |
| Fail（ratio < 0.5） | — | — | H_carryover 成立；pre-L state difference 是效應的必要部分；instruction span 在 L12–15 的 stance 搬運主要靠上游 carryover 累積 | Phase F 需先確認 carryover 來源：entity span L0–11 寫入的 pre-L state 是否直接 carryover 到 L12–15 的 instruction position |

---

## 8. 邊界與非目標

- Phase E 的 joint patch 與 2B full residual swap 的差異只有 `(pre_L_src − pre_L_tgt)` 項；E1 fail 不等於「instruction span 無效」，而是說明 pre-L carryover 是效應的必要部分。
- E2b 的 dial channel transplant 在 MLP 輸入空間（9216 維）操作，不是 residual 空間（2560 維）；e_{8490} ∈ R^{9216} 在此空間有效（channel index < 9216）。Footprint arm 在 residual 空間操作，有 confound，僅描述性。
- PCA 基底由 8 個 direction deltas 決定（±pair 結構 → rank ≤ 400，k-sweep {1,3,8,16,full} 非退化）；不對這 8 個 direction 的 population 外做推廣。
- `pca_basis_vectors` 是 derived statistic（從 direction deltas 計算，不是 raw hidden states），持久化符合 compact 政策。
- 16 家公司、Qwen3.5-4B、英文 prompt；不主張外推。
- bf16 jitter 下限 0.05 nats；joint ratio 在 |ΔM_joint| < 0.05 nats 時在 artifact 中標注。
- D1 的 Gate D fail 不影響 Phase E 的授權；Phase E 是獨立的鑑別實驗，不依賴 D1 pass。

---

## 9. Gate 授權鏈

```
Phase D smoke（已通過）
       ↓
Phase E smoke（需通過，驗收條件見 §4.1）
       ↓
Phase E formal run
   ├── Gate E1（dual-block joint 鑑別）
   └── Gate E2b（dial channel transplant carrier）；E2a descriptive
       ↓（依 Gate 結果，見 §7）
Phase F：carrier 確認或 carryover 溯源（另立協議）
```

---

## 10. 上游 Run 依賴

| 上游 Run | 路徑 | 用途 | 驗證方式 |
|---|---|---|---|
| `phase2a-gpu-bf16-01` | `artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01` | canonical prompt rows、pure entity margin、group gap pre-check | prepare SHA-256 + record 數 + manifest complete |
| `phase2b-gpu-bf16-01` | `artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01` | frozen 8 directions；2B L12/L15 T 作為 full-arm smoke 參照（±0.05 nats） | prepare SHA-256 + directions 集合相等 + manifest complete |

Phase D 的 run 記錄作為 context 引用（block-level null，smoke 已通過），不作為數值依賴。

---

## 11. 版本分立觸發條件

以下任一要素變更必須建新版本協議：patch 位置（instruction span → 其他）、joint patch 算術定義、layer 範圍（E1 L12–18、E2 L15）、Gate E1 門檻（0.5）、Gate E2b 門檻（0.5）、R1 denominator 門檻（0.2 nats）、PCA k sweep 集合（{1,3,8,16,full}）、dial channel（8490）、E2b 的 transplant 語義（position-restricted MLP-space → 其他）、公司 population、模型或 dtype。

---

## 12. Revision Record

| Rev | 日期 | 狀態 | 說明 |
|---|---|---|---|
| 0（Draft） | 2026-09-11 | superseded | 初稿：E1 joint ratio Gate + E2 PCA sweep + dial direction e_{8490} ∈ R^{2560}（空間錯誤）；basis 未持久化。 |
| 1 | 2026-09-11 | proposed | **B1 修正**：E2b 移至 MLP 原生空間，改為 position-restricted dial channel transplant（`δ(p) = a_src(p) − a_tgt(p)`）；加 residual footprint 描述性副臂。**B2 修正**：§2 PCA 基底改為右奇異向量（`V ∈ R^{2560}`）。**B3 修正**：持久化 top-16 右奇異向量（~400 KB）。**R1**：pre-register ratio denominator rule（`|ΔM_full| ≥ 0.2`）。**R2**：加 full-swap reference arm（in-run，bit-exact）；joint ratio 分母改用 in-run full T。**R3**：定義 smoke grid 與 5 項機制驗收條件。**R4**：移除「複用 2B state → 120 forwards」變體，統一 240 forwards（E1）；說明 joint 只使用 pre + post。**M1–M5**：§1.2 標題修正（D1 覆蓋 L12–31，最大 0.091 nats）；§2 joint state 定義改寫；§4.1 instruction span 對齊補充說明；record schema 加 `full_t_ref` 與 `dial_delta_per_position`；成本更新（~312 forwards，~15 min）。Tests 補充機器交叉驗證、dial position-restriction、R1 denominator rule 單測。 |
| 1.1 | 2026-09-11 | proposed | 實作紀錄（`joint_patch.py`、`dial_probe.dial_value_capture`、pipeline `run_phase_e` 四 stages、`scripts/entity_to_dial_phase_e.py`、`tests/test_entity_to_dial_joint.py` 22 個 fake-model 測試；全量 659 passed，4 個 pre-existing 失敗與本輪無關）。與 Rev 1 的實作差異：（a）E2 的 k=full 臂實跑 exact-swap forward（8 forwards）而非 Rev 1 計帳的 no-op 佔位（8 forwards）——重跑才能機器驗證驗收條件 #4「k=full ΔM == E1 full arm bit-exact」，佔位會讓 #4 變成空轉；E2 實際 80 forwards（4 k + full + full-noop + dial + dial-noop + footprint，per direction），Phase E 總計 320（非 §6 估算 312）。`pca_full_noop` 與 `dial_noop` 也實跑 self-source/空 delta forward 做 bit-exact 檢查。（b）smoke 驗收 #2 採 per-direction 對照（2B `records.jsonl` 本身含 per-direction `toward_source_delta_m`；in-run forward 確定性下應 bit-exact 對上存檔，±0.05 nats 容差吸收語義漂移），不是 8 方向平均對 §1.1 平均值。（c）`dial_noop` 用空 delta dict（hook 不註冊，bit-exact clean forward）。 |
| 1.2 | 2026-09-12 | proposed | 真模型 smoke：`entity-to-dial-e-smoke-20260912T085431Z`（2026-09-12，11 min）**PASS**——5 項機制驗收全過；E1 16 records（8 no-op 全 0.0）、E2 12 records（SVD 決定性、basis 16×2560 持久化）；**驗收 #2 的 in-run full arm 與 2B 存檔 per-direction bit-exact 一致（4/4 組合 diff = 0.0）**——cross-pipeline transform 語義與 forward 確定性在真模型上確認。第一次 smoke（`entity-to-dial-e-smoke-20260912T083841Z`）fail：`joint_patch` 的 delta rows 被 `.cpu()` 釘在 CPU，CUDA forward 時 device mismatch；修正為 transform rows 保留 forward device、apply 時 `.to(patched.device)` 防護、PCA/projection 在 CPU 計算（fake-model 測試全 CPU 未覆蓋，fake pipeline 測試補不回來，已記錄為 smoke 的價值）。 |
| 1.3 | 2026-09-12 | completed | Formal run `entity-to-dial-e-01`（2026-09-12，21 min 26 秒）完成，manifest complete、E1 224 records、E2 72 records、128 no-ops 全 0.0、全 finite；in-run full arm 與 2B 存檔 56/56 bit-exact（max\|diff\| = 0.0）。**Gate E1 pass**（L15 joint ratio median 0.575 ≥ 0.5，n_effective 6/8：R1 排除 BLK→BDX \|full ΔM\|=0.140 與 NSC→IT −0.097；6 個有效方向中 4 個比值 0.524–0.654，BLK→IT −1.391 與 NSC→BDX +2.353 為離群）——H_joint 勝出 H_carryover（L15 incremental_ref 0.1132 對應 carryover 預測比值 ≈ 0.187，實測 0.575 為其 3.1 倍；pre-L state difference 不是必要承載項）。**Gate E2b fail**（dial ratio median 0.434 < 0.5；falsified=False，未低於 0.05 falsifier 帶）——dial channel 是實質但未完整的 carrier。E2a descriptive：k=1 0.737 / k=3 0.885 / k=8 0.983（達 H_E2a target 0.8）/ k=16 1.001 / full 1.000——L15 state difference 低維集中（單方向 ≈ 74% 效應）。Footprint（descriptive）0.439 與 dial transplant 0.434 幾乎相同。讀法：§7 表「Pass / Fail（未 falsified）/ 低維」列——Phase F 以持久化 top PCA 方向做 single-direction validation，dial 作為部分 carrier 併入組合實驗。 |
