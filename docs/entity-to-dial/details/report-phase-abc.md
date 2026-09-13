# Entity-to-Dial 路徑解剖：Discovery 報告

**狀態**：completed（formal run 完成，2026-09-11；Gate A1 / Gate B / Gate C 皆 fail）  
**對象模型**：Qwen3.5-4B（bf16，hybrid Gated DeltaNet 架構）  
**協議**：[proposal.md](proposal-phase-abc.md)（Rev 1.3 frozen）  
**Run IDs**：entity-to-dial-a-01 / entity-to-dial-b-01 / entity-to-dial-c-01（皆 complete）  
**收線報告**：包含 Phase D/E/F 完整因果鏈與最終 L15 機制定位的總結見 [Entity-to-Dial 研究線收線報告](../report.md)（研究線已正式收線）

---

## Introduction：Entity 的決策偏誤路徑以三段式因果解剖定位

balanced-evidence-gap Phase 2B 確認了 entity identity 在 L0–11 的 entity span 殘差流中承載 entity-specific 決策影響（L0–5 normalized transfer T ≈ 1.0），並在 L12–15 交接至 instruction span（L15 峰值 T = +0.464）。同一條研究線的 Phase 3 排除了 L19/L20/L26 三個 MLP 神經元作為因果槓桿點（全部 |mean ΔM| ≤ 0.012 nats，低於 control noise floor）。另一端，investment-dial 線確認 L15/N8490 是模型層級的決策旋鈕（±4s 推注 → margin ±1.0–1.1 nats），但對所有公司施力方向一致，不認識特定 entity。

兩端已知，中間未解：entity 在 L0–11 寫入殘差流後，**哪些 token、哪幾層是充分的；L12–15 的 handoff 由 MLP 還是 attention block 承擔；entity bias 是否流經 L15/N8490 dial 座標**。本報告以三個 Phase 的 block-level 因果干預回答這三個問題。

---

## Methods：凍結 8 directions、block-level patch、FP32 tail margin

沿用 balanced-evidence-gap Phase 2A 的 16 家公司（4 sector × 4 家，test split）與 frozen shared-evidence template（2 正 2 負，公司中立）。8 directions 直接繼承 Phase 2B frozen pairs（NSC↔IT、NSC↔BDX、BLK↔IT、BLK↔BDX，雙向）；group gap 實測 1.028 nats（pre-check 門檻 0.5 nats）。

**Phase A**：在 L0–11 的 entity span 內，把 patch 範圍縮小到 ticker-group 與 name-group 兩個 token 子集，分別在 8 directions 下量 toward-source ΔM 與 normalized transfer T；整個 entity span patch 的 Phase 2B 存檔值作為上界參照。Gate A1：ticker-group 在 L0–5 的 mean ΔM 的 bootstrap 95% CI 排除 0。

**Phase B**：在 L12–15 對 entity span 位置分別替換 MLP block 貢獻（`mid + (post_source − mid_source)`）與 attention block 貢獻（`pre + (mid_source − pre_source)`），量 8 directions 的 toward-source ΔM。Gate B1：至少 1 層的 MLP block ΔM 的 CI 排除 0；Gate B3：最強層的 4 sector mean ΔM 同號。

**Phase C**：對 16 家 named/anonymous prompt 讀取 L15/N8490 dial activation（entity span 最後 token position），計算 per-company `δ = a_named − a_anon`；對 anonymous prompt 施加 `mlp_addition(layer=15, neuron=8490, delta=δ)` 推注，量 `ΔM_dial = M_pushed_anon − M_clean_anon`，對照 `gap = M_named − M_clean_anon`。Gate C1：mean ΔM_dial 的 CI 與 mean gap 同號；Gate C2：mean |ΔM_dial| / mean |gap| ≥ 0.25。

全程 FP32 tail-logit margin；self-source no-op（|ΔM| ≤ 1e-12）為 fail-closed 紀律；block arithmetic 在 FP32 計算後 cast 回 bf16；bf16 jitter 下限約 0.05 nats。

---

## Phase A：Ticker Token 在早期層的充分性

**Motivation**：L0–11 承載帶的 entity 訊號是否集中在 ticker symbol，以及 company name tokens 從哪幾層開始提供增量貢獻？

**Setup**：8 directions × L0–11 × {ticker-group, name-group} patch；Phase 2B 存檔 entity-span T 為上界參照（不重跑）。共 304 forwards。

**Findings**：

- Gate A1（ticker-group L0–5 mean ΔM CI 排除 0）：**fail**——mean ΔM = +0.159 nats，CI 95%：[−0.554, +0.828]（跨 0）
- Ticker-group L0–5：T = 0.09–0.15（per-layer CI 全部跨 0）；L6 後趨近 0（|T| ≤ 0.03）
- Name-group L0–5：mean ΔM = +0.912 nats，T = 0.68–0.77，per-layer CI 全部排除 0；顯著性延伸至 L10（L11 CI 開始含 0），T 緩降（L7–L10 = 0.47, 0.47, 0.39, 0.33）
- 2B entity-span upper-bound 參照 T（L0–5 平均）：+0.976——name-group 單獨解釋約 70–80%
- Per-direction 結構（描述性，非 gate 標的）：8 個 direction 的 raw margin 位移（patched − target）兩組全部朝 sell 方向——ticker raw ΔM（L0–5）= −0.67～−1.58（mean −0.98），name = −0.26～−2.23（mean −1.05）；patched margin 收斂於 anonymous baseline（m_anon ≈ −3.23，16 家幾乎常數）附近。bottom-source 方向的 T > 1（overshoot：patched margin 低於 source margin，掉向 baseline）；toward-source 統計主要由這 4 個 direction 驅動

**Interpretation**：H1（ticker 充分）不成立：ticker-group 在 L0–5 沒有顯著 transfer（T ≈ 0.1），顯著承載者是 company-name token（T ≈ 0.7，L0–10 皆顯著）。Entity 訊號在承載帶主要以 name token 的形式寫入，ticker symbol 不單獨承載 stance。另有一個值得後續驗證的描述性觀察：兩組的 raw 位移一致地朝 anonymous baseline 收斂（patched margin ≈ m_anon ± 0.4），而非「複製 source 的 stance」（那應在 8 個 direction 都產生 T ≈ 1）——即 L0–5 的 entity-position patch 表現像「抹除 entity 對齊後掉回無-entity 基準」，toward-source 統計主要由 bottom-source 方向的 overshoot 構成。此解讀為描述性，確認它需要一個 dedicated control（以 anonymous prompt 的 entity-position state 直接 patch），屬後續線。最後，本結論的主要不確定性來源：bottom-source 的 overshoot 與 top-source 的 undershoot 在 8 directions 中貢獻不對稱——若 erasure 機制成立，toward-source 統計是 m_src、m_tgt 與 m_anon 三者幾何關係的產物（以 erasure 預測值估算，8-direction mean T ≈ +0.5，與觀測 +0.70 同結構同量級），name-group T ≈ 0.7 的「transfer 效率」詮釋可能需要修正；name state 的 patch 產生 ~1 nat 位移這個效應本身是確立的（若 name state 不承載 entity 資訊，patch 應為 no-op），不確定的是該效應的性質（stance transfer vs. alignment erasure）。

---

## Phase B：L12–15 Handoff 的 Block 分工

**Motivation**：Phase 2C 用 first-order attribution 找到 L19/L20/L26 的相關訊號，Phase 3 排除了它們作為單神經元槓桿。L12–15 handoff 的實際承載者是 MLP block 還是 attention block？

**Setup**：8 directions × L12–15 × {mlp, attn} block-level patch，位置限定 entity span。共 144 forwards。

**Findings**：

- Gate B1（至少 1 層 MLP block ΔM CI 排除 0）：**fail**
  - Qualifying layers：無
  - MLP per-layer mean ΔM（L12–15）：−0.011, −0.041, −0.030, −0.059 nats（CI 全部跨 0，最寬 [−0.509, +0.484]）
- Gate B3（strongest layer，4 sector mean ΔM 同號）：未評定（B1 無 qualifying layer，strongest layer 未定義）
- Gate B：**fail**
- Attention block patch 對照（per-layer mean ΔM，L12–15）：−0.052, −0.052, −0.064, −0.071 nats（descriptive；與 MLP 同量級，皆在 0.05 nats bf16 jitter 帶附近或以下）

**Interpretation**：L12–15 的 entity position 上，MLP 與 attention block 各自的貢獻搬運都不產生顯著 margin 效應（|mean ΔM| ≤ 0.071 nats）。這與 2B 的 entity-span T 衰減一致（L12 = 0.26 → L15 = 0.03，同期 instruction span T 升至 L15 峰值 0.464）：L12 後 entity signal 的承載者已遷移至 instruction context，「handoff」不是 entity position 上的 block 分工，而是承載位置的切換。Phase 3 的單神經元 null（L19/20/26）也因此獲得粒度層面的支持：連整個 MLP block 搬運都無顯著效應，單顆神經元的 null 不是因為訊號分散在 block 內。

---

## Phase C：Entity Bias 流經 Dial 座標的比例

**Motivation**：entity A 的決策偏移是「修改 L15/N8490 激活值」的下游效果，還是走了 dial 旁邊另一條路？Phase 3 確認 dial 對照效應 ±1.0 nats，本 Phase 量 entity signal 借道 dial 的比例。

**Setup**：16 named + 16 anonymous clean forward，dial activation 讀取（entity position），16 pushed forward（`mlp_addition(15, 8490, δ)`）。共 49 forwards。

**Findings**：

- Step C1 描述性：dial_delta vs. pure entity margin Spearman ρ（entity position）：**0.159**（< 0.3 warning 門檻）；final position ρ = 0.403
- Phase 1 named-vs-anonymous gap（mean，16 companies）：`+0.432 nats`（Phase 1 存檔值；注意 Phase 1 使用公司別長文 evidence template，2A 起改用 frozen shared-evidence template，兩者 gap 不直接可比）
- 本 run gap（mean）：**+0.929 nats**（range +0.052～+1.708；IT 最小 +0.052），2A cross-check max |Δ|：**5.96e-08 nats**（無 warning）
- Gate C1（mean ΔM_dial CI 與 mean gap 同號）：**fail**——mean ΔM_dial = +0.013 nats，CI 95%：[−0.015, +0.041]（與 gap 同號但 CI 跨 0，非單邊）
- Gate C2（mean |ΔM_dial| / mean |gap|）：**0.051**（門檻 0.25，fail）
- Gate C：**fail**
- Step C3 unexplained gap（mean）：**+0.916 nats**（15/16 家為正；IT 為 −0.052）；dial_delta（a_named − a_anon，entity position）range −0.030～+0.075，mean +0.014

**Interpretation**：先交代位階背景：本 run 的 gap（+0.929 nats）是在 frozen shared-evidence（公司中立）template 下量的，比 Phase 1 的公司別長文 evidence template（+0.432 nats）大約 2.1 倍——證據模板公司中立後，company name 對 margin 的增量效應更大（公司別證據本身已攜帶公司資訊，name 的增量相對被稀釋）；兩者非受控比較（prompt 與量測時點皆不同），僅作為位階背景。在此 gap 上，entity 效應基本不流經 L15/N8490 dial 座標：named/anonymous 的 dial activation 差異（δ）只有 mean +0.014（raw activation units，range −0.030～+0.075），把 δ 推回 anonymous prompt 只產生 mean +0.013 nats 的 margin 位移（CI 跨 0），解釋 gap 的約 5%（C2 ratio 0.051 ≪ 0.25）。剩餘 ~95%（unexplained gap mean +0.916 nats）走 dial 以外的路徑。這與 investment-dial 線的結論（dial 是 model-level stance prior、對所有公司一致施力、不認識 entity）一致：dial 是平行的一般性槓桿，不是 entity→decision 路徑的中繼站。IT 的 gap ≈ 0.052 接近 0，其 per-company ratio 2.01 是近零分母的產物，不承載資訊。

---

## Discussion：三段解剖的整合結論

**跨 Phase 整合**：

- Phase A：entity 訊號在 L0–5 主要由 company-name token 承載（name T ≈ 0.7，ticker T ≈ 0.1），且 patch 的 raw 效應表現為「抹除 entity 後掉回 anonymous 基準」（描述性，待 dedicated control 確認）。
- Phase B：L12–15 的 entity position 上兩個 block 都無顯著貢獻；handoff 是承載位置從 entity span 切換到 instruction context（與 2B 的 L15 instruction peak 一致），不是 block 分工。
- Phase C：entity gap 只有約 5% 流經 dial 座標；主體路徑繞過 dial。entity→decision 路徑與 investment-dial 的 stance-prior 路徑是兩條平行路徑，不在 L15/N8490 會合。

**與既有線的整合**：

| 線 | 既有結論 | 本報告的連接點 |
|---|---|---|
| Entity Cell（L0–4） | 事實記憶與決策路徑功能解離（0 次翻轉） | Phase A 顯示 L0–5 entity 狀態的 patch 效應是「對齊抹除→掉回基準」而非 stance 搬運；entity cell 所在層（L0–4）的 entity 狀態同時承擔對齊功能，但其與事實記憶細胞是否同座標仍未驗證 |
| balanced-evidence-gap Phase 2B | L0–11 承載，L15 instruction peak | Phase A 把承載帶精化為 name-group（解釋 ~70–80% 的 entity-span T）；Phase B 確認 L12 後 entity position 不再承載，與 instruction peak 互為印證 |
| balanced-evidence-gap Phase 3 | L19/L20/L26 single-neuron null | Phase B 的 block-level null 說明 Phase 3 的 null 不是粒度問題：entity position 上整塊 MLP 搬運都無顯著效應，決策路徑不走這些座標 |
| Investment Dial（L15/N8490） | model-level stance prior，不認識 entity | Phase C 量化 entity 借道 dial 的比例 ≈ 5%（C2 ratio 0.051）：dial 路徑近乎虛無，entity→decision 路徑與 dial 平行 |

**後續方向（依實際結果）**：

- Phase C fail 且比例遠低於門檻（0.051 ≪ 0.25）：「繞過 dial」結論明確；unexplained gap（~95%）的路徑應從 L15+ 的 instruction context（2B peak 所在）到 final decision 區間的殘差幾何入手，需另立研究線。
- Phase B null：原設想的「MLP block 內部結構」線不成立為起點；block-level 解剖應改以 instruction span 的 L15+ 座標為對象。
- Phase A 的 entity-erasure 描述性觀察：確認實驗是用 anonymous prompt 的 entity-position state 直接 patch（預期 T ≈ 0 且 patched margin ≈ m_anon），需新版本協議。

---

## 限制

1. **bf16 jitter 下限約 0.05 nats**：FP32 tail margin 與 fp32 block arithmetic 消除了量測端的系統性誤差，但 forward 計算本身在 bf16 下存在跨 run jitter；任何 |ΔM| < 0.05 nats 的效應不作為強因果依據。
2. **Discovery 性質**：本報告為三個 Phase 的首次正式 run，不設 confirmation run；gate pass 的結論為 discovery，若需確認需另立研究線。
3. **Block-level patch 的語義邊界**：一階向量替換，不是 circuit-level 分離；Phase B 回答「哪個 block 的貢獻搬運後足以轉移 margin」，不排除 block 內部更細的結構（例如 MLP 的特定 neuron 子集）。
4. **Phase C 的 all-position 推注稀釋效應**：`mlp_addition` 對所有 token position 加同一 δ，而 entity signal 集中於 entity position；若 Gate C 邊界性 fail，entity-position-only 推注版本是優先的替代設計（需新版本協議）。
5. **16 家公司**：所有結論限於 2A population（Qwen3.5-4B，英文財報模板，4 sector × 4 家 test-split 公司）；不主張外推至 427 家宇宙或其他模型。

---

## Artifact 索引

| Phase | Run ID | Run root | 狀態 |
|---|---|---|---|
| A | entity-to-dial-a-01 | `artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-a-01/` | complete（gate A1 fail） |
| B | entity-to-dial-b-01 | `artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-b-01/` | complete（gate B fail） |
| C | entity-to-dial-c-01 | `artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-c-01/` | complete（gate C fail） |

Smoke runs（通過，§14）：`entity-to-dial-a-smoke-20260911T062930Z`、`entity-to-dial-b-smoke-20260911T063501Z`、`entity-to-dial-c-smoke-20260911T064217Z`。

---

## 附錄：Per-layer 完整曲線與 Per-direction 明細

正文只報告 gate verdict 與關鍵座標數字，完整曲線表於此。

### A. Phase A：Per-layer ΔM / T 曲線（ticker-group vs. name-group vs. 2B entity-span）

| Layer | ticker mean ΔM | ticker T | name mean ΔM | name T | 2B entity-span T（參照） |
|---|---|---|---|---|---|
| L0 | +0.209 | +0.146 | +0.897 | +0.701 | +1.011 |
| L1 | +0.159 | +0.108 | +0.863 | +0.676 | +0.957 |
| L2 | +0.135 | +0.087 | +0.914 | +0.718 | +0.987 |
| L3 | +0.155 | +0.104 | +0.877 | +0.684 | +0.982 |
| L4 | +0.158 | +0.104 | +0.943 | +0.741 | +0.971 |
| L5 | +0.138 | +0.090 | +0.980 | +0.770 | +0.945 |
| L6 | +0.023 | +0.014 | +0.952 | +0.745 | +0.856 |
| L7 | +0.036 | +0.022 | +0.623 | +0.475 | +0.627 |
| L8 | +0.042 | +0.026 | +0.626 | +0.474 | +0.569 |
| L9 | +0.009 | +0.003 | +0.509 | +0.394 | +0.482 |
| L10 | +0.000 | −0.004 | +0.436 | +0.334 | +0.459 |
| L11 | −0.011 | −0.009 | +0.400 | +0.311 | +0.441 |

### B. Phase B：Per-layer，MLP vs. Attention Block ΔM

| Layer | MLP mean ΔM | MLP CI | Attn mean ΔM | Attn CI |
|---|---|---|---|---|
| L12 | −0.011 | [−0.509, +0.484] | −0.052 | [−0.501, +0.399] |
| L13 | −0.041 | [−0.476, +0.403] | −0.052 | [−0.503, +0.401] |
| L14 | −0.030 | [−0.455, +0.413] | −0.064 | [−0.487, +0.364] |
| L15 | −0.059 | [−0.501, +0.382] | −0.071 | [−0.486, +0.347] |

### C. Phase C：Per-company gap / ΔM_dial / unexplained_gap

| Ticker | Sector | gap | ΔM_dial | unexplained_gap | dial_delta（entity pos） |
|---|---|---|---|---|---|
| BLK | Financials | +1.708 | +0.103 | +1.605 | +0.056 |
| AMAT | IT | +1.411 | −0.018 | +1.429 | −0.002 |
| NSC | Industrials | +1.327 | −0.017 | +1.344 | −0.009 |
| DE | Industrials | +1.311 | −0.016 | +1.328 | +0.013 |
| AXP | Financials | +1.277 | +0.057 | +1.219 | +0.030 |
| DHR | Health Care | +1.267 | +0.006 | +1.261 | +0.010 |
| GS | Financials | +1.094 | +0.025 | +1.069 | +0.034 |
| CSX | Industrials | +1.052 | −0.091 | +1.143 | −0.025 |
| ABT | Health Care | +0.909 | +0.046 | +0.863 | +0.024 |
| C | Financials | +0.877 | +0.100 | +0.777 | +0.054 |
| HON | Industrials | +0.772 | −0.011 | +0.784 | −0.001 |
| SYK | Health Care | +0.642 | −0.021 | +0.663 | −0.009 |
| GLW | IT | +0.522 | −0.080 | +0.602 | −0.030 |
| HPE | IT | +0.395 | +0.042 | +0.354 | +0.008 |
| BDX | Health Care | +0.247 | −0.027 | +0.274 | −0.002 |
| IT | IT | +0.052 | +0.104 | −0.052 | +0.075 |
