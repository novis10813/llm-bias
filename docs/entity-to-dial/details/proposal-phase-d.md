# Entity-to-Dial Phase D：Instruction Span Block Sweep 與 Signed Attribution 協議

**狀態**：completed（formal run `entity-to-dial-d-02` 完成，2026-09-11；Gate D fail：D1 無合格層；D2 描述統計：19/19 非 final 層 top channel |ρ| 勝過 matched controls，但 sector agreement 多數 0/4；首次 formal run `entity-to-dial-d-01` 的 D2 因 prepare rows 只含 4 家而無效，見 Rev 1.2）  
**對象模型**：Qwen3.5-4B（bf16，hybrid Gated DeltaNet 架構，32 層，full-attention 層為 L3/7/11/15/19/23/27/31）  
**研究線定位**：承接 [entity-to-dial discovery 報告](report-phase-abc.md) 的三個 Phase null 結果，以及 [balanced-evidence-gap Phase 2B](../../balanced-evidence-gap/details/proposal-phase2.md) 的 instruction span L15 峰值，定位 entity signal 在 instruction span 上的 block-level 承載層，並透過 signed attribution 為後續因果驗證（Phase E）提供 channel-level 候選。

---

## 1. 背景：三個 Phase Null 的共同含義

entity-to-dial Phase A/B/C 的結果組合如下：

| Phase | Patch 位置 | 結果 | 含義 |
|---|---|---|---|
| A | entity span，L0–11 | ticker-group T ≈ 0.1（null）；name-group T ≈ 0.7（顯著） | Entity signal 在早期層主要由 name token 承載；patch 效應表現為「掉回 anonymous baseline」而非 stance 搬運（描述性） |
| B | entity span，L12–15 MLP/attn block | 全 null（|mean ΔM| ≤ 0.071 nats） | L12 後 entity position 已不承載；handoff 是承載位置而非 block 分工 |
| C | L15/N8490 dial channel | ratio 0.051 ≪ 0.25 | Entity gap 只有 ~5% 流經 dial；主體路徑繞過 dial |

三個結果加上 Phase 2B 的 instruction span L15 peak（T = +0.464）共同指向同一個未解問題：**entity signal 在 L12 後存在於 instruction span 的殘差流，而非 entity position 或 dial channel。instruction span 上，哪幾層的 MLP/attention block 承載了 entity-conditioned 的決策信號？**

---

## 2. 新增術語定義

本協議沿用 [proposal.md §2](proposal-phase-abc.md#2-定義) 的術語。新增以下定義：

- **instruction_last_position**：instruction span 的最後一個 token position，即 `instruction_span[1] - 1`（0-indexed，與 2B 的 instruction span 定義一致）。所有 16 家公司共用 frozen shared-evidence template，instruction span 在 tokenized 後長度與最後 position 相同，1:1 aligned，不需要 nearest-position 映射。
- **signed MLP attribution**：在 instruction_last_position 對 MLP down-projection 輸入（activation `a`，shape `[9216]`）計算 `∂M/∂a`（保留正負號）。Phase 2C 用 unsigned `|∂M/∂a| · |a|`；本協議用 signed `∂M/∂a`，原因是符號直接決定推注方向（ρ > 0 的 channel → 正向推注使 margin 上升），不再需要額外的極性假設。
- **instruction-position attribution**：`∂M/∂a` 在 `instruction_last_position` 讀取的梯度向量，shape `[9216]`（per layer, per company）。與 Phase 2C 的 entity-position attribution（`|∂M/∂a| · |a|`，entity position 加總）形成對比：位置不同，signed vs. unsigned，不相互替代。
- **top channel**：某層的 `argmax_{k} |Spearman(∂M/∂a[:, k], pure_entity_margin)|`（Spearman ρ 的 k 為 channel index，跨 16 家公司計算）；ρ 的符號即為 predicted direction（ρ > 0 → 正向推注使 margin 上升）。
- **matched controls**（Phase D2）：與 2C 相同的 per-layer random sample，`random.Random(42 + layer).sample(range(9216), 10)`；Phase D2 不做 gate，controls 只用於描述性對比。

---

## 3. 假說

- **H_D1（instruction block 充分性）**：instruction span 上，L12–31 中至少 1 層的 MLP block patch toward-source ΔM 的 95% CI 排除 0，且最強層的 4 sector mean ΔM 同號（Gate D1 + D3）。
- **H_D2（instruction attribution，descriptive）**：L12–31 的 MLP 層中，instruction_last_position 的 signed ∂M/∂a 在某幾層有跨 16 公司顯著的 Spearman ρ（vs. pure entity margin），且方向一致（4/4 sector）；這些 channel 的 ρ 強度超過 matched controls。H_D2 為純 discovery，不 gate；輸出供 Phase E causal validation 使用。

---

## 4. 實驗設計

### 4.1 公司與 prompt 族

完全繼承 [Phase A/B/C §4.1](proposal-phase-abc.md#41-公司與-prompt-族)：16 家公司、8 directions（NSC↔IT、NSC↔BDX、BLK↔IT、BLK↔BDX 雙向）、frozen shared-evidence template、canonical variant（reverse=False, order=0）。

**Instruction span 1:1 對齊**：因為 template 對 16 家公司完全相同，instruction span 的 token sequence 逐位相同，block-level patch 為直接索引對應（identity mapping），不需要 nearest-position 啟發式——這是 Phase D 比 Phase B 設計更乾淨的地方，proposal 明確標注。

**Smoke pre-check**（同 Phase A/B）：
1. Group gap ≥ 0.5 nats（從 2A 存檔重算；已知 1.028 nats）。
2. Self-source no-op |ΔM| ≤ 1e-12（forward check）。

### 4.2 Sub-experiment D1：Instruction Span Block Sweep

**問題**：L12–31 的 instruction span 上，MLP block 與 attention block 哪幾層承載 entity-specific 決策信號？

**操作**：

- Layers：L12–31（20 層）。
- Components：MLP block output、attention block output（分別 patch，語義同 [Phase B §4.3](proposal-phase-abc.md#43-phase-b-handoff-區間-block-level-activation-patch)）。
- 位置：instruction span 所有 token position（identity mapping）。
- Directions：8 個（2B frozen pairs）。
- 每個 (layer, component, direction) 跑一次 patch forward，量 toward-source ΔM 與 normalized transfer T。
- 加 final position（1 × 2 components × 8 directions = 16 forwards）作為 sanity check（預期 T ≈ 0）。

**Forward 協議（per direction）**：

1. Source 狀態 forward：`record_block_states(source, L12–31)`（1 次）。
2. Target clean forward：`record_block_states(target, L12–31 + final layer)` + `instruction_last_position` 記錄（1 次），final-layer 走 FP32 tail 得 live target margin。
3. Patch forwards：20 layers × 2 components = 40 次。
4. Final position sanity：1 × 2 = 2 次。
5. Self no-op：20 layers × 2 components = 40 次；|ΔM − live target margin| ≤ 1e-12 fail-closed。

共 **8 × (2 + 40 + 2 + 40) = 672 forwards**。

**Gate D1**（pre-registered）：L12–31 中至少 1 層的 MLP block patch toward-source ΔM 的 bootstrap 95% CI（8 directions, n=2000, seed=42）排除 0。  
**Gate D3**（pre-registered）：Gate D1 qualifying layers 中 |mean ΔM| 最大的那層（strongest layer），4 sector mean ΔM 同號（4/4；sector mean 定義同 Phase B §4.3）。  
**Gate D pass = D1 + D3 通過。** MLP vs. attention block 的大小比較為描述性（不 gate）。

### 4.3 Sub-experiment D2：Instruction Position Signed Attribution

**問題**：L12–31 的 MLP 層中，instruction_last_position 的哪些 channel 的 signed ∂M/∂a 與 pure entity margin 在 16 家公司間有強 Spearman ρ？

**操作**：

- 對 16 條 named canonical prompt（reverse=False, order=0），每條做一次 **differentiable forward + backward**。
- 在 `instruction_last_position` 對 MLP down-projection 輸入（`dense_down_projection(model.layers[l]).weight` 的輸入 activation，shape `[9216]`）計算 `∂M/∂a`（signed 梯度向量），L12–31 的 20 個 MLP 層同時捕獲（單一 backward pass per prompt）。
- Per layer：`∂M/∂a` 是 shape `[9216]` 的向量；取 k = `argmax_k |Spearman(∂M/∂a[:, k], pure_entity_margin)|`（跨 16 家公司）作為 top channel；同時計算 10 個 matched controls 的 |Spearman ρ|（per-layer seed 42+layer）。
- 持久化：per layer 只存 `top_channel_idx`、`top_channel_rho`（signed）、`top_channel_sector_agreement`（4 sector 的 mean ∂M/∂a 同號比例）；controls 的 `max_control_rho`（10 個的 max |ρ|）；不存完整 9216 維梯度向量。

**Gradient 實作**：

- 複用 `mlm_bias.core.inference.mlp_addition.mlp_summed_derivatives` 的 hook 機制作為參照，但 D2 需要**per-position**梯度而非 all-position sum。在 `entity_to_dial` 的新模組 `attribution.py` 實作 `mlp_position_derivative(model, layers, position)` context manager：hook 改為只對 `activation[:, position, :]` 的梯度做 `collect`（不 `sum` over position dim）。hook lifecycle 複用 core 的 handle registry 慣例（exception-safe 移除）。不修改 core 的 `mlp_summed_derivatives`。
- **Gradient check（smoke 驗收）**：對 1 家公司 × 1 層，驗證 `∂M/∂a` 的 shape = `[9216]`、全 finite、相對於 `mlp_summed_derivatives`（all-position sum）的符號一致性（sign agreement ≥ 80%）。
- Frozen model（`model.eval()`、`torch.no_grad()` 關閉）：activation hook 在 `register_forward_pre_hook` 拿到的 tensor 可能無 grad；實作在 `not values.requires_grad` 時做 `detach().requires_grad_(True)`（與 `mlp_summed_derivatives` 相同），確保 backward 能通過該節點。Gated DeltaNet 層的 backward path 在 Phase 3 實測（`mlp_summed_derivatives` 已用於 L15/19/20/26 層）；L12–31 的所有 MLP 層結構相同，backward 可行性由 smoke gradient check 最終驗收。

共 **16 backward passes**（D2 與 D1 的 forward passes 分開跑：D1 在 no_grad 下，D2 在 grad mode 下）。

### 4.4 運行順序

```
prepare（provenance + rows）
→ forward_d1（no_grad，672 forwards）
→ forward_d2（grad mode，16 backward passes）
→ analyze（Gate D1/D3 + D2 descriptive）
→ finalize
```

D1 和 D2 在同一個 run 內串行，共用 `prepare` 與 `analyze`，分屬不同 forward function。

---

## 5. 實現與 Artifact

**Package**：`llm_bias/entity_to_dial/`（現有 package，新增子模組 `attribution.py`，不 import 其他 experiment package）。

新增子模組：

- `attribution.py`：`mlp_position_derivative(model, layers, position)` context manager（per-position signed 梯度捕獲）；`top_channel_stats(gradients, pure_entity_margins, controls_seed_base)` 純函式（Spearman ρ、top channel、sector agreement、control max |ρ|）。

**Operator**：`scripts/entity_to_dial_phase_d.py`（單一 operator，串行跑 D1 + D2）。

CLI 契約：

```bash
# formal run
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/entity_to_dial_phase_d.py \
    --model .cache/models/qwen3.5-4b \
    --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \
    --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \
    --run-id entity-to-dial-d-01

# smoke
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/entity_to_dial_phase_d.py \
    --model .cache/models/qwen3.5-4b \
    --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \
    --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \
    --smoke
```

**Run root**：`artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-d-<seq>/`（dataset slug `entity-to-dial`，與 Phase A/B/C 共用 dataset）。

**Stages**：`prepare` → `forward_d1` → `forward_d2` → `analyze` → finalize（`required_stages = {"prepare", "forward_d1", "forward_d2", "analyze"}`）。

**Artifact schema（compact）**：

- `prepare/rows.jsonl`：16 行 named canonical prompt（繼承 Phase C schema，加 `instruction_last_position: int`）；`prepare/provenance.json`（上游 run SHA-256 + record 數 + manifest status）。
- `forward_d1/records.jsonl`：`{phase: "d1", direction, layer, component ∈ {mlp, attn, final_mlp, final_attn, self_noop_mlp, self_noop_attn}, patched_margin, toward_source_delta_m, normalized_transfer, m_source, m_target, live_target_margin}`。
- `forward_d2/records.jsonl`：per-layer 跨公司聚合（top channel 為跨 16 家 Spearman，非 per-company 統計）：`{layer, top_channel_idx, top_channel_rho, top_channel_sector_agreement, n_sectors, control_channel_idxs, max_control_rho, n_companies, vector_finite: true}`；不存 9216 維梯度向量。（draft 的 `ticker` 欄位與 `gradient_finite` 名稱不適用：前者與跨公司聚合語義衝突，後者觸發 core serializer 的 `gradient` 保留字，改名 `vector_finite`。）
- `analyze/summary.json`：`gate_d1`（qualifying layers、strongest layer idx、mean ΔM、CI、pass）；`gate_d3`（sector means、pass）；`gate_d`（D1 AND D3）；`d1_curves`（per layer × component 的 mean ΔM、T、CI）；`d2_descriptive`（per layer：top channel idx/ρ、sector agreement、max control ρ；D2 不 gate）；`final_position_sanity`（final position 兩 component 的 mean T）；`smoke: bool`。
- `manifest.json`（全部 artifact SHA-256 + record counts + stage lifecycle）。

**持久化**：不存 raw activations、residuals、梯度向量或 hidden states。D2 只存 per-layer 的 top channel 標量統計。

**Tests**（`tests/test_entity_to_dial_attribution.py`，fake model + mocked backward）：

- `mlp_position_derivative`：shape 正確（`[9216]`）、position 選擇正確（只捕獲指定 position 的梯度、其他 position 不影響）、exception-safe hook 移除、`requires_grad` 路徑（detach + requires_grad_）。
- `top_channel_stats`：Spearman 計算、argmax 選取、sector agreement（4/4 邊界、含 0 邊界）、control max |ρ|。
- gradient check（小型真實 `Qwen3_5ForCausalLM`，2–4 層）：shape、finite、sign agreement ≥ 80%（vs. `mlp_summed_derivatives`，限同層 position 加總後對比）。
- pipeline smoke path（monkeypatched）：D1 + D2 串行，manifest complete，schema 欄位齊全，no-op enforcement。

---

## 6. 成本估算

| Sub-experiment | Forwards / Backward | 說明 | 估算 |
|---|---|---|---|
| D1（block sweep） | 672 fwd | 8 × (2 capture + 40 patch + 2 final + 40 no-op) | ~15 min |
| D2（signed attribution） | 16 backward | 16 companies × 1 pass，捕獲 L12–31 × 9216 梯度（轉換成標量後立即釋放） | ~5 min |
| **Phase D 合計** | **~688** | 單次模型載入 | **~20–25 min** |

---

## 7. Gate 授權鏈

```
Phase A/B/C smoke（已通過）
       ↓
Phase D smoke（需通過，含 gradient check）
       ↓
Phase D formal run
   ├── Gate D1 + D3（D1 block sweep）
   └── D2 descriptive（不 gate）
       ↓（若 Gate D pass）
Phase E：instruction-position channel causal validation（另立協議）
```

Gate D1/D3 的判準和門檻 frozen 後不得修改；若需調整建新版本協議（不回填）。Gate D fail = 有效結果，run 仍 finalize 為 complete。

---

## 8. 邊界與非目標

- Phase D 不重開 Phase A/B/C 的任何 gate，不修改 entity-cell 或 investment-dial 的既有結論。
- D1 的 instruction span patch 是整段 token 的 block 貢獻替換，不是 per-token 精確定位；若 D1 pass，instruction span 內部的 token-level 充分性是後續研究（類比 Phase A 對 entity span 的精化）。
- D2 只存 top channel 標量，不存完整梯度向量；若 Gate D pass 後需要 per-channel 分佈，另立版本重跑（需授權）。
- D2 的 signed ∂M/∂a 是 local first-order sensitivity，不是因果效應；top channel 結論為 discovery，Phase E 才做 causal validation。
- Gated DeltaNet 層（L12–14 等線性注意力層）的 attention block 在 D1 中指該子層對殘差流的整體貢獻替換，非 full-attention 權重。
- 16 家公司、Qwen3.5-4B、英文 prompt；不主張外推。
- bf16 jitter 下限 0.05 nats；|ΔM| < 0.05 nats 的效應在報告中標記。
- D1 Gate D pass 不等於「instruction span MLP block 是唯一路徑」；attention block 的 descriptive 數字提供並行路徑的上界參照。

---

## 9. 預期知識收益

| Gate D | D2 top channel ρ | 後續 |
|---|---|---|
| Pass | > 0.7，sector 4/4 | Phase E：instruction-position top channel 的 `mlp_addition` causal validation（held-out 設計，另立協議） |
| Pass | 0.3–0.7 | Phase E：考慮擴大候選 channel pool（top-k）或 joint intervention |
| Fail | 任意 | entity bias 在 instruction span 上不靠單一 block 承載；需轉向整層 residual patch（L12–31 full-layer sweep）或更細粒度（per-token instruction patch） |

---

## 10. 上游 Run 依賴

| 上游 Run | 路徑 | 用途 | 驗證方式 |
|---|---|---|---|
| `phase2a-gpu-bf16-01` | `artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01` | canonical prompt rows、pure entity margin（D2 Spearman 依據）、group gap（pre-check 1） | prepare SHA-256 + record 數 + manifest complete |
| `phase2b-gpu-bf16-01` | `artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01` | frozen 8 directions（directions.json） | prepare SHA-256 + directions 集合相等 + manifest complete |

Phase A/B/C 的 run 記錄作為 report 的 context 引用，不作為 Phase D 的 upstream 依賴（不從 entity-to-dial-a/b/c-01 讀取數值）。

---

## 11. 版本分立觸發條件

以下任一要素變更必須建新版本協議：patch 位置（instruction span → 其他）、layer 範圍（L12–31）、attribution 方法（signed ∂M/∂a → unsigned 或 ×|a|）、attribution 讀取 position（instruction_last_position → 其他）、Gate D1/D3 判準或門檻、matched controls seed 規則、公司 population（16 家）、模型或 dtype。

---

## 12. Revision Record

| Rev | 日期 | 狀態 | 說明 |
|---|---|---|---|
| 0（Draft） | 2026-09-11 | proposed | 初稿：D1 instruction span block sweep（L12–31）+ D2 signed attribution（instruction_last_position）；單一 operator 串行；Gate D1/D3 pre-registered；D2 純 discovery。 |
| 1 | 2026-09-11 | proposed（實作完成） | 實作完成：`llm_bias.entity_to_dial.attribution`（`mlp_position_derivative`、`differentiable_margin`、`top_channel_stats`）、pipeline `run_phase_d`（prepare → forward_d1 → forward_d2 → analyze）、operator `scripts/entity_to_dial_phase_d.py`、tests（`tests/test_entity_to_dial_attribution.py`＋gate D 加入 `tests/test_entity_to_dial_analysis.py`）。Schema/語義調整（均不改 gate 判準）：(a) `forward_d2` record 的 `gradient_finite` 改名 `vector_finite`（core serializer 將 `gradient` 列為 raw-payload 保留字，同 Phase C `residual_gap`→`unexplained_gap` 先例）；(b) `forward_d2` record 為 per-layer 跨公司聚合（top channel 本就是跨 16 家的 Spearman 統計），draft 的 `ticker` 欄位不適用，移除；(c) `top_channel_stats` 簽名加 `sector_of`（sector agreement 所需）與明示的 `controls_n`/`controls_seed_base`；(d) instruction span「1:1 對齊」在實測 2A 數據中是**等長但絕對位置偏移**（header 長度不同；16 家皆 100 tokens，start 114–119）：forward_d1 用 offset-preserving nearest-position mapping（start 相同時退化為 identity），prepare fail-closed 驗證 16 家 span 等長並記錄 `instruction_span_length`；(e) final-position sanity 在 final layer（L31）以 `{target_final: source_final}` mapping（final position 因 prompt 長度而逐公司不同）；(f) final layer 的 D2 梯度在 non-final position 上**結構性為零**（final layer 的 block output 在該 position 之後不被任何計算讀取，margin 只讀 final position）：smoke sign check 對 final layer 記錄 `structural_zero` 並跳過，formal run 該層 record 的 ρ 依構造為 0；(g) smoke 網格：`SMOKE_D_LAYERS=(12,15,31)`、D2 smoke 公司 = NSC（1 家）。 |
| 1.1 | 2026-09-11 | proposed（smoke 通過） | **D2 smoke 驗收重新錨定**（不改任何 gate 判準）：Rev 0/1 的「position 梯度 vs. all-position 總和 sign agreement ≥ 0.8」在首次真實模型 smoke（`entity-to-dial-d-smoke-20260911T085517Z`，D1 部分通過、no-op 全 0）上 fail（L12 實測 0.519、L15 0.479）——0.8 門檻假設單一 position 的梯度主導 all-position 總和，但實測 position 貢獻是分散的（非主導），且 bf16 跨 run 低位元 noise 使 sign 在近平衡 channel 上不可靠；該門檻是對數據的錯誤預測，不是實作缺陷的訊號。驗收改錨在機制本身的數學不變量（fail-closed，同一 smoke 階段）：(a) **結構性零**：final layer 在 non-final position 的梯度嚴格為 0（final layer 的 block output 在該 position 之後不被任何計算讀取，margin 只讀 final position；首次 smoke 前於真實模型驗證為 True）；(b) **connectivity**：non-final 層梯度非零；(c) **partition identity**：all-position 捕獲的 fp32 position 加總與 core `mlp_summed_derivatives`（Phase 3 已驗證實作）的 norm-relative 誤差 ≤ 0.05（真實模型實測 2.1e-3）。Rev 0 的 sign agreement 仍計算並記錄（descriptive，不 gate）。新增 `mlp_all_positions_derivative`（attribution.py）供 (a)–(c) 使用。 |
| 1.2 | 2026-09-11 | proposed（第一次 formal run 部分無效，修正後重跑） | **d-01 D2 協議偏差＋修正**：第一次 formal run（`entity-to-dial-d-01`，47m56s，pipeline 層面 complete）的 `prepare` rows 只含 4 個 direction tickers（與 Phase A/B 同式），導致 D2 只對 4 家跑 backward（協議 §4.3 要求 16 家全 2A 人口做 cross-company Spearman）。**D1 結果有效**（D1 只需 source/target 4 tickers；Gate D1/D3 判定不受影響：fail，無合格層）；D2 描述統計無效。修正：`prepare` rows 改為全部 16 家 canonical 公司（D1 以 lookup 取 source/target，行為不變；smoke grid 與 no-op 語義不變），以 `entity-to-dial-d-02` 重跑。d-01 保留作為偏差紀錄。 |
| 1.3 | 2026-09-11 | completed | **Formal run 完成**（`entity-to-dial-d-02`，50m22s；D1 與 d-01 bit-exact 重現）。**Gate D fail**：D1 無合格層（L12–31 全部 8-direction toward-source ΔM bootstrap CI 跨 0；層間無結構，mean T ≈ −0.02 至 −0.09）；但 per-direction raw ΔM 一致為 sell 方向（−0.48 至 −0.66 nats，遠離 0.05 nat jitter 帶）且 toward 值近完美反對稱（如 BDX->BLK +0.484 / BLK->BDX −0.641）——與 Phase A 相同的方向性取消結構，erasure 簽章在 block 層級重現。2B 的 L15 instruction peak（whole-residual patch，T=0.464）在 block-level patch 未重現（L15 為最弱層之一，mean T=−0.022）。final-position sanity：兩 component mean T ≈ −0.11（小幅一致效應，descriptive）。**D2 描述統計（16 家）**：19/19 非 final 層的 top channel |ρ|（0.953–0.987）皆高於 matched control 最大 |ρ|（0.642–0.937）——channel 層級存在可辨識的 margin-tracking 結構；但 top channel 符號跨層不一致（多數 −）、sector agreement 多數 0/4（L15 2/4、L27 4/4），無單一層級可定位為「dial 座標」。L31 全部為零（結構性零 invariant 再次驗證）。 |
