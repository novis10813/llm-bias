# C2 v3：在 steering prompt 上的跨公司 span × layer patching（R6）與生成驗證（R7）

**狀態：**事前協議，2026-09-25，smoke 前凍結。**上層協議：**[confirmation-v1](../confirmation-v1/proposal.md)。**Claim：**C2（steering site 可由 layer 與 span 定位）。不改寫 [C2 v2-427](../c2-v2-427/status.md) 或 BEG 的凍結 `SPAN_NAMES`；新的 span 定義放在 `llm_bias/core/steering/prompts.py`。

## 為何需要 v3

C2-427 用 BEG pos↔neg 的同公司條件翻轉、固定前綴 readout，而 steering 評估用 balanced prompt、生成決策。v3 直接在 steering prompt 上做跨公司 patching，只用 construction 公司，並以與生成路徑一致的 readout 定位。

## 設計（`arm_c2v3`）

- 方向：construction 上的乾淨固定前綴 margin 最高 20 家（由高到低）與最低 20 家（由低到高）按名次配對，每對雙向，共 40 個方向；受測 101 家不參與。
- Span：`entity`、`evidence`、`instruction`、`final`（最後一個 prompt token）、`answer_prefix`、`steer_suffix`（instruction 最後 K 個 token）。對應規則與 BEG Phase 2 相同：entity 用 nearest-normalized mapping，其餘等長 span 用 offset identity、不等長時用 nearest；patch 在該層 block 輸出以 source 殘差取代 target 殘差。
- 層：全部層。
- **Primary readout：teacher-forced clean-path margin。**把每家公司自己的 α0 生成（token ids）接在 prompt 後、截到決策值 token 之前，在該位置取 `log p(buy-token) − log p(sell-token)`（token pair 取自 α0 生成的 realized readout）。source 殘差也在 source 自己的 teacher-forced 序列上擷取；`answer_prefix` 為生成的決策值之前的 token。任一端 α0 無法 realized 時，該方向只有 secondary readout。
- **Secondary readout：**固定前綴 margin（prompt ＋ `{"decision": "`），`answer_prefix` 為這段固定前綴。
- `T = (M_patched − M_tgt) / (M_src − M_tgt)`，`M_src`、`M_tgt` 為各自的乾淨 margin（同一 readout）；|M_src − M_tgt| < 1e-6 時 T 記 None。
- 曲線：各 (readout, span, layer) 的平均 T，方向層級 bootstrap（B=2000）95% CI。
- **Band 規則（凍結）：**teacher-forced `steer_suffix` 曲線上 `T ≥ 0.7 × peak` 的層；無 teacher-forced 值時改用固定前綴曲線並標記。
- 檢查：self-patch（source = target）在所有層與 span 上固定前綴 margin 差 ≤ 1e-6（smoke 失敗即停，full 記錄）。

## R7：patch-under-generation（`arm_c2v3_gen`）

- 層：R6 peak 與 peak ± 2；span：`steer_suffix`、`entity`；40 個方向；patch 只作用在 prefill（decode 步略過）。
- 指標：source 與 target 的 α0 決策不同的方向中，patched 生成決策等於 source α0 決策的數量（toward-source flips），加 parse 數；另報 patched 生成的 realized-path margin 朝 source 乾淨 realized margin 方向的平均位移（2026-09-25 smoke 後補入：Qwen 的 α0 幾乎全為 sell，toward-source flip 沒有分母）。
- 10 家 self-patch 生成必須與 α0 逐字相同。

## C2 可寫範圍與 A2

- 可寫：「以 construction-only、steering renderer、預先登記規則選出的 patching band，包含主要層的 DIM 效應（容許 ±2 層）」；raw 劑量下的 depth containment 不算證據（depth sweep 已刪除）。「entity L0–5」只有在 R6 的跨公司 entity span 重現時才保留。GPT-OSS 只作描述。
- A2（`scripts/summarize_c2_overlap_exclusion.py`，CPU）：從 C2-427 records 中排除 101 家受測公司的方向後重算 instruction 曲線與 band，並報只含重疊公司的曲線，與登錄表的 peak／band 並列。

## 成本

每模型 40 × 層數 × 6 × 2 次 forward（Qwen 約 1.5 萬、Gemma 約 2.3 萬）＋ R7 約 40 × 3 × 2 ＋ 10 列生成，另加 40 家 α0。
