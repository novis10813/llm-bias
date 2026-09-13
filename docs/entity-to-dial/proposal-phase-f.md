# Entity-to-Dial Phase F：Dual-Path Additivity 與 Directional Push 協議

**狀態**：completed（Rev 1.4；formal run `entity-to-dial-f-02`，Gate F1 fail，F2 兩臂 context-dependent/null；依 §7 fallback 以 k=8 子空間版本收線）  
**對象模型**：Qwen3.5-4B（bf16，hybrid Gated DeltaNet 架構，32 層，full-attention 層為 L3/7/11/15/19/23/27/31；hidden_size = 2560，intermediate_size = 9216）  
**研究線定位**：承接 Phase E（`entity-to-dial-e-01`，Gate E1 pass / Gate E2b fail）的發現——L15 instruction span 的 entity state difference 由兩條近乎正交的效應路徑承載（residual 空間 v₁ 分散式方向 73.7%、MLP 空間 dial channel 8490 43.4%，cos(v₁, dial footprint) = −0.020）——以 dual-hook 同時介入驗證兩路徑的加法性，並以 neutral-context directional push 判定 v₁ 的 stance loading 結構（signed axis vs. attractor vs. context-dependent）。本協議不重開 Phase A–E 的任何 gate，也不修改既有結論。

---

## 1. 背景：Phase E 結果與兩條路徑的幾何關係

### 1.1 Phase E 的主要量測（L15 instruction positions，8 directions）

| 介入 | 空間 | effect ratio（有效方向中位數，E 的 R1 convention） |
|---|---|---|
| Full residual swap（in-run，與 2B 存檔 56/56 bit-exact） | residual | 1.0（參照） |
| PCA k=1（v₁ 投影 transplant） | residual | 0.737 |
| PCA k=3 | residual | 0.885 |
| PCA k=8 | residual | 0.983 |
| PCA k=16 | residual | 1.001 |
| Dual-block joint（E1） | residual（block delta 組合） | 0.575 |
| Dial channel transplant（position-restricted） | MLP input | 0.434 |
| Dial residual footprint（W_down[8490] 方向投影） | residual | 0.439 |

### 1.2 事後幾何掃描（e-01 持久化 top-16 右奇異向量 + 模型權重，無 GPU）

- `cos(v₁, dial footprint) = −0.020`；**全部 16 個 PCA 基底向量對 dial footprint 的 |cos| < 0.071**——dial 的 residual 方向在 8-direction state-difference 子空間之外。
- v₁ 對 L15 全部 9216 個 down-proj input channel footprint 的最大 |cos| = 0.298（channel 8324）：**v₁ 是分散式 residual 方向，不共線於任一單一 channel**，無法用 dial 的 channel 定位路線解讀。
- Dial footprint 對其他 channel 的最大 |cos| = 0.267（channel 8482）：dial 的 residual 方向基本不與其他 channel 共用。
- Dial 在 9216 個 channel 中對 v₁ 的 |cos| 排名 3258/9216（中等偏無特色）。

### 1.3 Per-direction 結構：stance transfer asymmetry

| direction | full | v₁ (k=1) | k=8 | dial | joint |
|---|---|---|---|---|---|
| BDX→BLK | +1.324 | +0.71 | +0.97 | +0.36 | +0.54 |
| BDX→NSC | +0.947 | +0.76 | +0.98 | +0.59 | +0.65 |
| IT→BLK | +1.439 | +0.71 | +0.99 | +0.33 | +0.52 |
| IT→NSC | +1.065 | +0.78 | +1.02 | +0.51 | +0.61 |
| BLK→IT | +0.274 | −0.20 | +0.93 | −2.15 | −1.39 |
| NSC→BDX | −0.258 | +1.88 | +1.04 | +2.53 | +2.35 |

（toward-source ΔM，nats。BLK→BDX +0.140、NSC→IT −0.097 因 R1 規則排除。）

- 4 個 bottom→top（賣方 stance source）方向全部強轉移；4 個 top→bottom（買方 stance source）方向全部弱或負。
- 兩個有效 top→bottom 方向上 v₁ 與 dial 都把 margin 推向 sell（與 toward-source 反向），full swap 的溫和結果由 pre-L state difference 抵消（brake）：BLK→IT 的 full − joint = +0.274 − (−1.39) ≈ +1.66 nats（buy 方向）。
- 描述性讀法：**sell stance 是可轉移的 attractor，buy stance 不是**；pre-L difference 在 top→bottom 方向扮演 brake 而非被動背景。此讀法需要 neutral-context push 實驗確認（Phase F2）。

### 1.4 未解問題

1. **加法性**：v₁ 與 dial 分屬 residual 空間與 MLP 空間，未曾在同一 forward 同時介入。若兩路徑獨立加性，combined 應 ≈ v₁ + dial；若下游匯流（shared downstream readout），combined 會飽和。注意 k=8（0.983）已幾乎飽和 full swap，而 dial footprint 與全部 16 個 PCA 向量近乎正交——dial 的 43.4% 與 PCA 子空間的貢獻在 state 幾何上不相交，但下游效應可能匯流；dual-hook 實驗是唯一判別。
2. **v₁ 的 loading 結構**：per-direction 表顯示 v₁ 的效應符號在 top→bottom 方向反轉（push sell），但 transplant 實驗的係數是 `(Δs·v₁)`（含 direction-specific 符號），無法直接讀出 v₁ 方向本身的 stance loading。Neutral prompt 上 `±αv₁` 的對稱性測試是唯一乾淨的判別。

---

## 2. 新增術語定義

沿用 [proposal.md §2](proposal.md#2-定義)、[proposal-phase-d.md §2](proposal-phase-d.md#2-新增術語定義)、[proposal-phase-e.md §2](proposal-phase-e.md#2-術語定義) 的術語。新增：

- **v₁ / k 維 PCA 子空間**：Phase E e-01 持久化於 `analyze/summary.json` 的 `pca_basis_vectors`（top-16 右奇異向量，shape [16, 2560]，FP32，orthonormal）。v₁ = 第 1 個向量；k=8 子空間 = 前 8 個向量張成的空間。Phase F 直接從 artifact 載入（SHA-256 provenance 檢查），不重算。
- **v₁ transplant**：對 instruction positions P，`post_L15_tgt[:, P, :] += (Δs·v₁) v₁`（per-position，FP32 arithmetic，cast 回 bf16）；與 E2-2 k=1 arm 算術相同。
- **k=8 transplant**：`post_L15_tgt[:, P, :] += V₈ V₈ᵀ Δs`（V₈ = 前 8 個右奇異向量）；與 E2-2 k=8 arm 算術相同。
- **dial transplant**：L15 MLP down-projection 輸入的 channel 8490，instruction positions p 各加 `δ(p) = a_src(p) − a_tgt(p)`（per-position，其他 positions 與 channels 不動）；與 E2-3 arm 算術相同。
- **dual-hook combined patch**：單一 forward 內同時註冊兩個 hook——（a）L15 post-block 點的 residual transform（v₁ 或 k=8 transplant）；（b）L15 MLP down-projection 輸入的 dial transplant。實作為巢狀組合 context manager：core `residual_interventions`（layer forward hook，post-block）套 `dial_channel_transplant`（down-proj forward_pre_hook）——兩 hook 掛在不同 module，語義與 Phase E/E2b 的既有路徑 bit-exact 相同（不重新實作 hook）；composite 外加 residual transform 的 strict single-fire 計數與 fire-count box 供 smoke 驗證；exception-safe 移除由兩層巢狀 context manager 各自保證。smoke 必須驗證 dual-hook no-op bit-exact 0.0 與「單 hook 置零 = 單 arm」的組成性質。
- **additivity ratio**：per-direction `ΔM_combined / ΔM_full`（有效方向定義與 median 規則**完全繼承 Phase E E2b 的 ratio convention**，同一套程式路徑，不另立定義）。Gate F1 的標的。
- **additive residual**：per-direction `ΔM_combined − (ΔM_v1 + ΔM_dial)`（descriptive）。≈ 0 表示兩路徑一階獨立加性；顯著為負表示下游飽和或匯流。
- **directional push**：對 neutral（anonymous）prompt 在 instruction positions P 加**constant-across-positions** 的向量 `±α × push_base × d`（d 為 unit direction，α ∈ {0.5, 1.0, 2.0}）。與 transplant（per-position Δs 投影）語義不同：push 測試方向 d 在 neutral context 的本征 loading，不依賴任何 source entity state。
- **push base**：`base = median`（8 個 direction）of [`median`（P 個 instruction positions）of `|Δs_dir(p)·v₁|`]（residual activation units；雙層中位數：先 per-direction 對 positions 取中位數，再對 8 個 direction 取中位數；從 F1 的 in-run capture 計算）。α = 1.0 即「典型 v₁ 分量幅值」；dial footprint arm 與 v₁ arm 共用同一 base。
- **dial footprint direction**：`W_down[8490, :]`（L15，shape [2560]）歸一化，residual 空間 unit direction（E2-4 arm 同義）。
- **stance attractor**（描述性術語，不 gate）：cross-entity state transplant 把 margin 推向 sell 方向的現象在 buy-stance source 方向也出現（top→bottom 方向的反向 push），即 sell 方向對介入具有吸引性、buy 方向沒有對稱效應。F2 的 `±α` 對稱性測試是其 falsifier。

---

## 3. 假說與 Gate

### Gate F1（pre-registered）：dual-path additivity

- **H_F1**：dual-hook combined patch（v₁ transplant + dial transplant 同時）的 additivity ratio 在有效方向（R1 convention，分母為 in-run full arm）的 median ≥ 0.85。
- 判讀：pass = 「v₁ + dial 兩通道緊湊表示」能恢復 ≥ 85% 的 full swap 效應（與 k=8 的 0.983 同一量級，但只用 1 個 residual direction + 1 個 MLP channel）；fail = 兩路徑存在下游飽和/交互，緊湊表示不成立。
- **Secondary statistic（pre-registered，descriptive）**：4 個 bottom→top direction（group 成員 2A frozen：bottom = {IT, BDX}，top = {NSC, BLK}）的 additivity ratio median。Gate 用全部有效方向的 median（與 E2b 同 convention），secondary 用於解讀 stance asymmetry。

### H_F2（descriptive，不 gate）：v₁ 的 stance loading 結構

F2 的 `±α` push 在 neutral context 讀出 v₁（與 dial footprint 對照）的 loading 結構。pre-registered 判讀規則見 §4.3。

**Gate F pass = Gate F1 pass**（F2 為 discovery 讀數，不影響 gate verdict）。

---

## 4. 實驗設計

### 4.1 公司與 prompt 族

- 8 directions 與 16 家公司完全繼承 Phase A–E（2A population，2B frozen pairs）。
- F2 的 neutral prompt：`entity_to_dial.spans.anonymous_prompt`（Phase A/C 既有建構）對 canonical 2A prompt 做單一 byte-level header 替換（`Stock Ticker: [<TICKER>]` / `Stock Name: [<Name>]` → `Stock Ticker: [TICKER]` / `Stock Name: [Company X]`，其餘 template 不變）；instruction span 由 `prompt_char_spans` 的 instruction 字元區間 + `token_span` 映射（與 2A `resolve_row` 同一機制）。**Prepare 步驟必須驗證 16 家公司的 anonymous prompt 字串完全相同**（frozen template + 統一佔位，預期 token-identical）；若不同 → fail-closed，需新版本協議。記錄該單一字串的 SHA-256 於 `prepare/provenance.json`。Phase C c-01 存檔已確認 16/16 家 anonymous margin 同值（−3.2280，token-identical 的實證）。
- **v₁ / PCA 基底 provenance（fail-closed）**：從 `entity-to-dial-e-01/analyze/summary.json` 載入 `pca_basis_vectors` 與 `pca_singular_values`；驗證 manifest complete、summary SHA-256 記錄、16 向量 orthonormal（atol 1e-6）、singular values 遞減。任一失敗 → 中止。
- Smoke pre-check：group gap ≥ 0.5 nats（2A 存檔重算）。

### 4.2 Sub-experiment F1：Dual-Hook Combined Patch（L15）

**Arms（per direction）**：

| Arm | 算術 | 用途 |
|---|---|---|
| full | exact swap `post_L15_src`（2B bit-exact 算術） | in-run 分母 + 跨 run 一致性檢查（vs 2B / e-01） |
| v₁ | v₁ transplant | in-run 複現 E2 k=1 |
| k8 | k=8 transplant | in-run 複現 E2 k=8（**saturation 參照**） |
| dial | dial transplant | in-run 複現 E2 dial |
| combined | dual-hook（v₁ + dial 同時） | **主要量測** |
| 每 arm 的 no-op | 零 delta / self-source | fail-closed，預期 bit-exact 0.0 |

**State capture（per direction，2 forwards）**：source 與 target 各一次 `record_block_states(L15)` + dial pre-hook capture `a_{8490}(p)` + target 的 FP32 tail margin（live）。Capture 後在記憶體內派生：`Δs_dir`（[P, 2560] FP32）、per-position `δ_dir(p)`、`push_base`（8 direction 完成後）。Transient states 在 derive 後釋放，不持久化。

**Forward 協議**：8 directions × (2 capture + 5 arms + 5 no-op) = 8 × 12 = **96 forwards**。

**跨 run 一致性檢查（pre-registered，descriptive）**：in-run v₁ / k8 / dial / full arm 與 e-01 E2 k=1 / k=8 / dial 與 2B 存檔的 per-direction ΔM 差異 ≤ 0.01 nats（同模型同 GPU 決定性；e-01 的 full arm 已 bit-exact）。差異 > 0.01 → smoke 不通過，查因重跑。

**Gate F1**：additivity ratio（§2 定義，E2b convention）在有效方向 median ≥ 0.85。

**Per-direction 完整表（descriptive）**：6 個有效 direction 的 full / v₁ / k8 / dial / combined 五列 ΔM + ratio + additive residual；4 個 bottom→top 與 2 個 top→bottom 分組 median 各列。

### 4.3 Sub-experiment F2：Neutral-Context Directional Push

**Prompt**：1 條 anonymous prompt（§4.1 驗證 16 家相同字串；取 NSC-derived row 作 provenance 錨點）。Clean forward × 1（m_anon 基準，預期 ≈ −3.23 nats）。

**Arms（per prompt）**：

| Arm | Push 向量（positions P，constant across positions） | 點數 |
|---|---|---|
| v₁ push | `±α × base × v₁`，α ∈ {0.5, 1.0, 2.0} | 6 |
| dial_fp push | `±α × base × (W_down[8490,:] / ‖·‖)` | 6 |
| no-op | 零 push | 1 |

**Forward 協議**：1 clean + 12 push + 1 no-op = **14 forwards**。

**量測**：`ΔM = M_pushed − M_clean_anon`（FP32 tail margin）。

**Pre-registered 判讀規則（v₁ arm；dial_fp arm 同法）**：以 α ∈ {1.0, 2.0} 的 4 個點（+v₁ × 2、−v₁ × 2）判定：

| 觀測 | 分類 | 含義 |
|---|---|---|
| +v₁ 兩點皆 ΔM < −0.05 且 −v₁ 兩點皆 ΔM > +0.05 | signed sell axis | v₁ 是 signed stance 軸，+ 方向 = sell；entity 偏誤 = v₁ 軸上的 loading 差 |
| +v₁ 兩點皆 ΔM > +0.05 且 −v₁ 兩點皆 ΔM < −0.05 | signed buy axis | v₁ 是 signed stance 軸，+ 方向 = buy；§1.3 的 sell-attractor 讀法需修訂 |
| +v₁ 與 −v₁ 同方向（皆 sell 或皆 buy） | attractor | v₁ 編碼 deviation magnitude 而非 signed stance；stance attractor 成立（方向由觀測決定） |
| 混合 / 全部 \|ΔM\| ≤ 0.05 | context-dependent / null | v₁ 的效應依賴 transplant context（target state 互動），非本征 loading 方向 |

α = 0.5 的點用於 dose-response 形狀（線性 vs. 飽和）descriptive 記錄，不進判定。|ΔM| ≤ 0.05 nats 的點標 jitter-band。

### 4.4 運行順序

```
prepare（provenance + rows + anonymous identity check）
→ forward_f1（8 directions × 10 forwards）
→ forward_f2（anonymous push，14 forwards）
→ analyze（Gate F1 + per-direction 表 + F2 判定 + 一致性檢查）
→ finalize
```

F1 與 F2 在同一 run 串行；F2 的 `push_base` 依賴 F1 capture 的結果，順序固定。

---

## 5. 實現與 Artifact

**Package**：`llm_bias/entity_to_dial/`，`joint_patch.py` 擴展（Phase E 實作基礎上）：

- `make_projected_transform(delta, basis_k, positions)`：per-position `V_k V_kᵀ Δs` 投影 transplant（新 low-level API：接受預計算的 per-position Δs rows + basis + positions；k=1 即 v₁ transplant，k=8 即 k8 transplant；與 E2 `project_delta` 的 k=1 結果 bit-exact 等價，unit test 驗證。Phase E 的 high-level `make_projected_transform(source_post, target_post, basis, mapping)` 維持不變）。
- `make_dial_transplant(...)`：position-restricted channel transplant（E2-3 已實作，抽出共用）。
- `dual_hook_interventions`（`make_dual_hook`）：巢狀組合 core `residual_interventions` + `dial_channel_transplant`（見 §2 定義）；residual transform 包 strict single-fire 計數，回傳 fire-count box 供 smoke 驗證；任一 forward 例外不洩漏 hook（兩層各自 exception-safe）。
- `load_pca_basis(e01_summary_path)`：fail-closed 載入 + orthonormal/遞減/finite 驗證（§4.1）。
- `directional_push_transform(push_vector, positions)`：constant-across-positions 的 residual push transform（F2 用）。

**Operator**：`scripts/entity_to_dial_phase_f.py`（單一 operator，串行 F1 → F2）。

CLI 契約：

```bash
# formal run
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/entity_to_dial_phase_f.py \
    --model .cache/models/qwen3.5-4b \
    --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \
    --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \
    --phasee-run artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-e-01 \
    --run-id entity-to-dial-f-01

# smoke
... --smoke
```

**Run root**：`artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-f-<seq>/`。

**Stages**：`prepare` → `forward_f1` → `forward_f2` → `analyze` → finalize（`required_stages = {"prepare", "forward_f1", "forward_f2", "analyze"}`）。

**Artifact schema（compact）**：

- `prepare/rows.jsonl`：8 directions 的 named pairs + 1 anonymous row（含 instruction positions）；`prepare/provenance.json`（2A / 2B / e-01 三上游 SHA-256 + manifest status + anonymous 字串 SHA-256 + PCA 基底驗證結果）。
- `forward_f1/records.jsonl`：`{phase: "f1", direction, arm ∈ {full, v1, k8, dial, combined, *_noop}, patched_margin, toward_source_delta_m, additivity_ratio (combined 臂), m_source, m_target, noop: bool}`。
- `forward_f2/records.jsonl`：`{phase: "f2", arm ∈ {v1, dial_fp, noop}, sign ∈ {+1, −1}, alpha, push_norm, clean_margin, pushed_margin, delta_m, jitter_band: bool}`。
- `analyze/summary.json`：`gate_f1`（per-direction ratio、有效方向 median、pass）；`f1_secondary`（bottom→top / top→bottom 分組 median，descriptive）；`f1_per_direction`（完整 6×5 ΔM 表 + additive residual）；`f1_consistency`（vs e-01 / 2B 的 max |Δ|）；`f2_verdict`（v₁ / dial_fp 各自的 §4.3 分類 + 4 點 ΔM 表）；`f2_dose_response`（α 曲線）；`push_base`；`smoke: bool`。
- `manifest.json`（SHA-256 + record counts + stage lifecycle）。

**持久化**：不存 raw activations、`Δs` 向量或 dial per-position delta 向量；PCA 基底不重新持久化（引用 e-01 的 SHA-256 即可）。

**Smoke grid**（mechanism validation，不評 gate）：directions {BDX→BLK}（1 個 strong bottom→top）+ anonymous prompt；arms：full、v₁、k8、dial、combined + 5 no-op + F2 的 v₁ / dial_fp 各 ±α=1.0（4 pushes）+ 1 clean + 1 no-op。

驗收（全部必過）：
1. 全部 no-op ΔM = 0.0 bit-exact（**含 dual-hook combined no-op**）。
2. **Dual-hook 組成性質**：combined 臂在 dial δ = 0 時 bit-exact 等於 v₁ 臂；在 v₁ Δ = 0 時 bit-exact 等於 dial 臂（零-delta 變體機器驗證）。
3. In-run v₁ / k8 / dial / full 與 e-01 / 2B 的差異 ≤ 0.01 nats。
4. PCA 基底 provenance 檢查（§4.1）通過。
5. Anonymous 16 字串 identity 驗證通過；clean m_anon finite 且 |m_anon − (−3.23)| ≤ 0.1 nats（Phase A 參照）。
6. F2 push margins finite、|ΔM| ≤ 5 nats。

**Tests**（`tests/test_entity_to_dial_phase_f.py`，fake model + mocked forward）：

- `make_projected_transform`：k=1 與 E2 `project_delta` 的 bit-exact 等價；self-source（Δs = 0）→ no-op bit-exact；k=8 投影冪等性（`V₈V₈ᵀV₈V₈ᵀ = V₈V₈ᵀ`）。
- `make_dual_hook`：單 hook 置零 = 對應單臂（bit-exact）；兩 hook 同時 no-op → margin bit-exact 等於 clean；exception-safe 移除（forward 中途 raise 後無殘留 hook）。
- `directional_push_transform`：constant-across-positions（P 內所有 position 加同一向量）；P 外 position 不變；零 push → no-op。
- `load_pca_basis`：orthonormal 失敗 / singular values 非遞減 / SHA 不匹配三個 fail-closed 路徑。
- Gate F1 邏輯：E2b ratio convention 的繼承（同一函式呼叫）、0.85 邊界、有效方向過濾；secondary 分組 median。
- F2 判定邏輯：§4.3 四種分類各一個合成 case；|ΔM| = 0.05 邊界歸 jitter-band。
- Pipeline smoke path：F1 + F2 串行，manifest complete，schema 欄位齊全。

---

## 6. 成本估算

| Sub-experiment | Forwards | 說明 | 估算 |
|---|---|---|---|
| F1 | 96 | 8 × (2 capture + 5 arms + 5 no-op) = 8 × 12 | ~7 min |
| F2 | 14 | 1 clean + 12 push + 1 no-op | ~1.5 min |
| **Phase F 合計** | **110** | 單次模型載入，串行 F1→F2 | **~10 min** |

（Phase E 實測 312 forwards / 21 min 26 s ≈ 4.1 s/forward；110 forwards ≈ 7.5 min，加載入與 finalize 緩衝取 10 min。）

---

## 7. 讀法與後續決策

| Gate F1 | F2 v₁ 判定 | 含義 | 後續 |
|---|---|---|---|
| Pass（≥ 0.85） | signed sell axis | 兩路徑緊湊表示成立（1 residual direction + 1 MLP channel ≥ 85%）；v₁ 是 signed stance 軸（+ = sell）；entity 偏誤 = v₁ 軸 loading 差 + dial 的平行貢獻 | **L15 機制定位完成**：entity→decision 的 L15 段可用「v₁ 軸位置 + dial activation」兩通道描述；收線報告的核心。可選後續：v₁ 的上游 write point（L12–13 寫入層）與 sell-attractor 的機制來源，另立研究線 |
| Pass | attractor | 緊湊表示成立；v₁ 編碼 deviation magnitude，sell 方向有吸引性、buy 沒有；§1.3 的 stance transfer asymmetry 獲直接確認 | 同上收線；attractor 結構寫入收線報告作為機制描述 |
| Pass | signed buy axis | 緊湊表示成立，但 §1.3 的 sell-attractor 讀法錯誤：v₁ 的 + 方向是 buy，top→bottom 的反向 push 來自非線性而非 v₁ 本征 loading | 修訂 §1.3 解讀後收線 |
| Pass | context-dependent / null | 緊湊表示成立，但 v₁ 無本征 loading：transplant 效應來自 target state 互動 | 收線時標注 v₁ 的效應是 context-dependent；「stance 軸」描述降級為「state-difference 主方向」 |
| Fail（< 0.85） | 任意 | 兩路徑下游飽和/匯流：combined < 加性預期；additive residual 的負值即交互強度 | 緊湊表示不成立；fallback 是 k=8 子空間（0.983，E2 已量測）作為 L15 段的完整描述；收線報告採用 k=8 版本 |

注意：Gate F1 pass + 任意 F2 判定 = 收線條件成立（L15 機制定位完成）；F2 只影響收線報告的機制描述，不影響 gate。若後續要追「為什麼 buy 不可轉移」（brake 的來源層、write point），是獨立的新研究線，不在本協議。

---

## 8. 邊界與非目標

- Phase F 只介入 L15（Phase E 鑑別出的最強層）；L12–14 不重測，e-01 records 作參照。
- v₁ 與 k=8 子空間由 8 個 direction deltas 決定（±pair 結構，rank ≤ 400）；不對這 8 個 direction 的 population 外做推廣。
- Dual-hook combined patch 是兩路徑的**同時一階介入**，不是 circuit-level 分離；additivity 成立不等於兩路徑在計算圖上不相交。
- F2 的 push 是 constant-across-positions（同一向量加在 P 內所有 position），與 F1 的 per-position transplant 語義不同；兩者的絕對數值不可直接比較，只有符號與劑量形狀可比較。
- F2 的判定門檻（±0.05 nats）等於 bf16 jitter 帶；貼邊結果在 report 中標 jitter-band 不強判。
- Anonymous prompt 的 16 字串 identity 是 prepare 的 fail-closed 檢查；若 template 未來改版導致不同，本協議的 F2 需新版本。
- 16 家公司、Qwen3.5-4B、英文 prompt；不主張外推。

---

## 9. Gate 授權鏈

```
Phase E formal run（已完成：entity-to-dial-e-01，Gate E1 pass / E2b fail）
       ↓
Phase F smoke（需通過，驗收條件見 §5 六項）
       ↓
Phase F formal run
   ├── Gate F1（dual-hook combined additivity ratio ≥ 0.85）
   └── F2 descriptive（v₁ / dial_fp loading 判定）
       ↓（依 §7）
收線報告（L15 機制定位完成；機制描述依 F2 判定分派）
或 fallback 分析（Gate F1 fail → k=8 子空間版本收線）
```

---

## 10. 上游 Run 依賴

| 上游 Run | 路徑 | 用途 | 驗證方式 |
|---|---|---|---|
| `phase2a-gpu-bf16-01` | `artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01` | canonical prompt rows、group gap pre-check、stance group 成員（bottom/top 分組） | prepare SHA-256 + record 數 + manifest complete |
| `phase2b-gpu-bf16-01` | `artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01` | frozen 8 directions；2B L15 T 一致性參照 | prepare SHA-256 + directions 集合相等 + manifest complete |
| `entity-to-dial-e-01` | `artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-e-01` | **PCA 基底**（`pca_basis_vectors`，F1 的 v₁/k8 臂與 F2 的 v₁ push 方向）；E2 k=1/k=8/dial records（一致性檢查參照） | manifest complete + summary SHA-256 + orthonormal/遞減/finite 驗證 |

Phase A/B/C/D 的 run 記錄作為 context 引用，不作為數值依賴。

---

## 11. 版本分立觸發條件

以下任一要素變更必須建新版本協議：介入層（L15 → 其他）、combined 臂的 hook 組合（v₁ + dial → 其他，例如 k8 + dial）、Gate F1 門檻（0.85）、R1 convention（0.2 nats 有效方向規則）、F2 的 α 集合（{0.5, 1.0, 2.0}）、push base 定義（median |Δs·v₁| → 其他）、F2 的 neutral prompt（anonymous template 改版）、direction population（8 directions）、PCA 基底來源（e-01 → 其他 run）、模型或 dtype。

---

## 12. Revision Record

| Rev | 日期 | 狀態 | 說明 |
|---|---|---|---|
| 0 | 2026-09-11 | superseded | 初稿：F1 dual-hook combined patch（v₁ + dial 同時，L15，8 directions；arms = full / v₁ / k8 / dial / combined + no-op；Gate F1 additivity ratio ≥ 0.85，E2b convention；k8 臂作 saturation 參照）+ F2 neutral-context directional push（anonymous prompt，±α × {0.5, 1.0, 2.0} × median|Δs·v₁|，v₁ 與 dial footprint 兩臂；pre-registered 四分類判定，descriptive 不 gate）；PCA 基底從 e-01 持久化 artifact 載入（fail-closed provenance）；anonymous 16 字串 identity fail-closed 檢查；~94 forwards，~10 min；單一 operator 串行。 |
| 1 | 2026-09-12 | proposed | **B1（forwards 帳算術）**：§4.2/§6 的 8 × (2+5+5) 為 96（非 80），Phase F 總計 110 forwards（~10 min 不變）。**B2（push base 定義）**：`median_dir |Δs_dir·v₁|` 是 per-direction P 向量，改為雙層中位數標量：先 per-direction 對 P 個 positions 取 `median |Δs(p)·v₁|`，再對 8 個 direction 取中位數。**B3（anonymous 建構 pinned）**：明確重用 `entity_to_dial.spans.anonymous_prompt`（Phase A/C 既有；`Stock Ticker: [TICKER]` / `Stock Name: [Company X]` 佔位）+ `prompt_char_spans`/`token_span` 映射 instruction span；Phase C c-01 存檔確認 16/16 家 anonymous margin 同值 −3.2280（token-identical 實證），in-run m_anon 預期 bit-exact 重現，驗收 #5 的 −3.23 ± 0.1 帶維持。**B4（dual-hook 語義）**：巢狀組合 core `residual_interventions` + `dial_channel_transplant`（不重新實作 hook；兩 hook 語義與 E2b 路徑 bit-exact 相同）+ strict single-fire 計數 box。**B5（API 簽名）**：新 low-level `make_projected_transform(delta, basis_k, positions)`（接受預計算 Δs rows）；Phase E 的 high-level 簽名維持不變。 |
| 1.1 | 2026-09-13 | 實作備註 | 實作落點與本協議的細部對齊（不改變任何 gate/門檻/方向 population）：**C1（B5 命名）**：low-level API 實作為 `make_projected_transplant(delta, basis, positions)`——`make_projected_transform` 名稱已被 Phase E high-level 簽名（`source_post, target_post, basis, mapping`）佔用，同 module 無法以 B5 要求的方式重載，改新名避免衝突；k=1/k=8 算術與 bit-exact 等價性驗證不變。**C2（PCA 基底 in-memory 約定）**：`load_pca_basis` 回傳 `[d, k]`（右奇異向量為欄，與 Phase E `pca_state_directions` 的 in-memory 約定一致）；持久化格式維持 e-01 的 `[k, d]` rows；orthonormal 檢查在持久化 rows 上做（FP64 row-Gram，atol 1e-6）。**C3（record 欄位）**：F1 records 在 §5 schema 外另含 `normalized_transfer`、`live_target_margin`、`noop_delta_m`、dial 臂的 `dial_delta_per_position`（與 Phase E record 風格一致）；F2 no-op 臂名 `noop`。**C4（一致性檢查位置）**：驗收 #3 在 smoke forward 階段 inline 執行（fail-closed，tolerance 0.01）；formal run 的跨 run 差異以 `f1_consistency`（per-arm max |Δ| + tolerance）記錄於 summary，descriptive 不 fail。**C5（summary 欄位對照）**：§5 的 `f1_per_direction` 6×5 表 = `gate_f1.per_direction`（full/v1/k8/dial/combined ΔM + ratio + additive residual）；F2 的 4 點 ΔM 表 = `f2_dose_response`（完整 α grid {0.5, 1.0, 2.0} × ±）。**C6（fire count）**：dual-hook context manager yield `{"residual": int, "dial": int}`；smoke 對 combined 臂斷言 (1, 1)；strict single-fire 在任一侧第二次觸發時 raise。 |
| 1.2 | 2026-09-13 | smoke 完成 | Smoke run `entity-to-dial-f-smoke-20260913T024304Z`（10 min 56 s，manifest complete），六項驗收全過：（1）5/5 no-op ΔM bit-exact 0.0（含 dual-hook combined_noop）；（2）組成性質 bit-exact：combined_dialzero +0.9392 ≡ v1 臂、combined_v1zero +0.4716 ≡ dial 臂；（3）in-run 與 e-01 一致性 full/v1/k8/dial = 1.3237/0.9392/1.2830/0.4716，diff 0.0；（4）PCA 基底 provenance 通過（16×2560 orthonormal atol 1e-6、singular values 3.506→1.183 遞減）；（5）anonymous 16/16 字串 identity（SHA-256 `53b658a8…`）、m_anon = −3.22799（Phase C 錨點 −3.2280，帶內）；（6）F2 push ΔM 皆 finite、max |ΔM| = 0.046 ≤ 5。描述性（smoke 方向 BDX→BLK 單方向，非 gate）：combined toward = +0.916（additivity ratio 0.692）< v1 +0.939，additive residual −0.495——該方向顯示非加性跡象，formal 以 6 個有效方向中位數判定；push_base（單方向）= 0.0054，四支 α=1.0 push 皆在 jitter 帶內（|ΔM| ≤ 0.05，描述性）。formal run 待授權。 |
| 1.3 | 2026-09-13 | 首次 formal 失敗 + key 更名 | 授權後的首次 formal run `entity-to-dial-f-01` 在 analyze 階段 fail-closed：core artifact serializer 拒絕 summary key `additive_residual`（key 的 `_` 分段含保留字 `residual`；smoke 未觸發因 smoke gate = not_evaluated 不持久化 per-direction 表）。forward_f1（8 directions 完整記錄）與 forward_f2（14 forwards 完整記錄）皆已寫入並通過當階段驗證；run 保留於 `artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-f-01/`（manifest status = failed，作為偏差記錄）。**C7（key 更名）**：`gate_f1.per_direction` 的 additive residual 量改以 JSON key `interaction_delta_m` 持久化（文件術語「additive residual」不變；算術不變：ΔM_combined − (ΔM_v1 + ΔM_dial)）；新增 unit test 對 formal gate 結構做 core 保留字守衛的遞迴檢查（防再犯）。重跑為 `entity-to-dial-f-02`（同一協議 Rev 1，無 gate/門檻/設計變更）。 |
| 1.4 | 2026-09-13 | formal 完成 | Formal run `entity-to-dial-f-02`（12 min 14 s，manifest complete；與 f-01 的 F1/F2 records bit-exact 相同，確定性確認）。**Gate F1 = fail**：6/8 有效方向（NSC→IT、BLK→BDX 依 R1 排除）的 additivity ratio median = 0.7329 < 0.85。Per-direction：4 個 bottom→top 方向 ratio 0.6920/0.7673/0.6985/0.7875，additive residual 全為負（−0.4949/−0.5500/−0.4908/−0.5364）——一致的下游飽和；2 個有效 top→bottom 方向符號混雜（NSC→BDX ratio +1.9560、BLK→IT −0.2775；兩者 additive residual 皆正 +0.6351/+0.5674，即 combined 皆比兩臂線性和「較不極端」，同樣指向匯流）。Secondary：bottom→top median 0.7329 / top→bottom median 0.8393。Consistency vs e-01：full/v1/k8/dial 四臂 diff 0.0（bit-exact 重現）。**F2 = 兩臂皆 context_dependent_or_null**：push_base（雙層中位數，8 directions）= 0.00540 residual units；v₁ 判定點（α=1.0/2.0）+push +0.0060/+0.0126、−push −0.0457/+0.0036（α=2.0 反號），dial_fp +push −0.0136/−0.0134、−push −0.0004/+0.0237——8 個判定點全部在 ±0.05 jitter 帶內。m_anon = −3.22799（錨點 −3.2280，帶內）。**依 §7 決策表**：Gate F1 fail + 任意 F2 判定 → 「緊湊表示不成立；fallback 是 k=8 子空間（0.983，E2 已量測）作為 L15 段的完整描述；收線報告採用 k=8 版本」。機制解讀（描述性）：v₁（residual）與 dial（MLP channel）上游幾何獨立（cos = −0.020）但下游 readout 匯流/飽和，combined 在各方向皆小於兩臂線性和；v₁ 在 neutral context 無可偵測的本征 signed loading，transplant 效應係 target-state 互動（context-dependent）；「stance 軸」描述降級為「state-difference 主方向」。 |