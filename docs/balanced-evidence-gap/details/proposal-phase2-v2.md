# Balanced Evidence Gap — Phase 2 v2：427 家公司條件翻轉實驗（development）

**狀態**：development（非 protocol-final；不回填 v1 結果）。v1 協議
（[proposal-phase2.md](proposal-phase2.md)）維持 frozen；本文件只定義
v2 的偏離項與 v2 run 的語義。
**對象模型**：Qwen3.5-4B（bf16，GPU0）。
**動機**：v1 16 家公司 universe 下，4B 的 pure entity margin 全數為負
（無一家 buy 方向），「buy → sell」panel 實際上是最 sell-leaning ↔
最 sell-leaning，語義偏弱。v2 改用**同一公司內部的證據條件翻轉**
（positive 條件 ↔ negative 條件），讓 buy/sell 兩端由同一個模型在
同一公司上自己定義，並在 427 家公司的 scale 上取得每層
mean ± standard deviation 的 direction 分布。

---

## 1. 設計

### 1.1 Universe

`data/baseline/investment-dial/exploratory-v1.json` 的全部 427 家公司
（只用 `ticker` 與 `name`；dataset 的 per-company `evidence_pairs`
**不使用**，v2 沿用 v1 的 shared-evidence 句）。

### 1.2 Prompt family（v2，兩句同向證據條件）

與 v1 的 cross-entity probe 骨架逐字相同（header、evidence markers、
JSON instruction tail、span 定義），唯一差異是 evidence block 只放
**兩句**同向 shared 證據，形成每個公司兩個條件：

| 條件 | 證據句 |
|---|---|
| `pos` | `EVIDENCE_P1` + `EVIDENCE_P2`（v1 frozen 正面兩句） |
| `neg` | `EVIDENCE_N1` + `EVIDENCE_N2`（v1 frozen 負面兩句） |

Variant 軸：保留 reverse（選項順序 `"buy" or "sell"` / `"sell" or
"buy"`）給 framing 控制；句子排列軸不存在（同向兩句只有一個 canonical
順序）。每公司 2 條件 × 2 reverses = 4 prompts；全 universe
427 × 4 = 1708 forwards。

實作：`template.build_prompt_v2` / `spans.resolve_row_v2`（row 增加
`condition` 與 `key = <ticker>:<condition>` 欄位）。

### 1.3 2A 分析與 gate

- 每公司每條件 margin = 該條件 2 個 reverse variants 的 median。
- **condition difference**：`D_t = M_pos(t) − M_neg(t)`。
- Gate 2A（reference-free；Qwen Phase 1 Spearman 參考與 L15/n8490
  dial readout 對 v2 不適用，強制關閉）：
  - IQR：`IQR({D_t}) > 0.5`（v2 語義：對 condition difference 的
    分散度，而非 v1 的 pure entity margin 分散度）。
  - framing stability：各條件內 reverse-pair delta 的 median |Δ| < 1.5。
  - schema validity：1.0。

### 1.4 2B：within-company condition-flip directions

- 每家公司貢獻 2 個 direction：`(t:pos → t:neg)` 與 `(t:neg → t:pos)`。
- Source/target 的 canonical row 各為該條件 reverse=False 的 row；
  patch 契約與 v1 相同（residual-stream resample、FP32 tail-logit
  margin、self-source no-op 每 (layer, span) 驗證）。
- `T = (M_patched − M_tgt) / (M_src − M_tgt)`；`|M_pos − M_neg| <
  0.1 nats` 的公司無法定義 T（分母近零），skip 並記入 run metadata
  的 `skipped_directions`。
- Sweep：L0–L31 × 4 spans（entity / evidence / instruction /
  final）。主觀測 span 是 **instruction**（沿用「只 swap instruction
  不 swap entity」的設定）；其餘 span 為 protocol completeness
  保留。v2 下 entity span 兩側文字相同（同公司），evidence span
  文字不同，position mapping 走 `nearest_position_mapping`。
- 427 家 × 2 directions × 32 layers × 4 spans ≈ 109k records
  （另加 self-source no-op forwards）。

### 1.5 圖

兩 panel：**左 = 全部 pos→neg directions（427）**，**右 = 全部
neg→pos directions（427）**；每層畫 direction-level mean T +
shaded standard deviation band，x 軸為相對深度 ℓ/(n_layers−1)。
matplotlib skill 樣式（whitegrid、despine、dimgrey ticks、
transAxes 註解、PDF+PNG）。

---

## 2. 與 frozen v1 的偏離項

| 項 | v1（frozen） | v2（development） |
|---|---|---|
| universe | 16 tickers（4 sector × 4） | 427 tickers（investment-dial 全量） |
| 每 prompt 證據 | 4 句混合（2 正 2 負） | 2 句同向（pos / neg 條件） |
| variant 軸 | 2 orders × 2 reverses | 1 order × 2 reverses |
| direction 構造 | top-2 vs bottom-2 跨公司 | 同公司 pos ↔ neg 條件翻轉 |
| gate 2A IQR 對象 | pure entity margin | condition difference |
| dial / Phase 1 Spearman | 適用（Qwen） | 強制關閉 |
| 2A/2B CLI | 預設 | `--family v2`、`--companies-file`、`--direction-min-gap` |

v1 的 artifact schema、frozen 常數、public CLI 行為皆未改動；v2 以
附加 flag 與 `V2_SCHEMA_VERSION = balanced-evidence-gap-phase2-v2`
區分。

## 3. Run 計劃與狀態

| run | 模型 | GPU | run-id | 狀態 |
|---|---|---|---|---|
| v2 2A | Qwen3.5-4B | 0 | `phase2a-v2-427-01` | running（runner `scripts/downloads/run_v2_427_qwen4b.sh`） |
| v2 2B | Qwen3.5-4B | 0 | `phase2b-v2-427-01` | queued（同 runner 串接） |

## 4. 語義邊界

- 固定答案 token 的 margin 位移只證明內部讀出信號改變；本實驗沒有
  定義真實生成（non-fixed-answer）的 decision-flip 指標，因此不主張
  決策行為改變。
- 2B 的 T 是 first-order transfer 統計量，不是 standalone causal
  proof；self-source no-op 僅驗證 patching 機制自洽。
- v2 結果屬 development，不回填 v1 文件、不進 frozen protocol 敘事。
