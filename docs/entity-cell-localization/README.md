# Entity Cell Localization and Downstream Attribution: Versions & Phases

本頁是 entity-cell 研究線（`llm_bias/entity_cell/`，CLI `entity-cell`）的文檔入口。依據 [`docs/AGENTS.md`](../AGENTS.md) 規範，本研究線解耦為三個獨立階段，每個階段各自維護獨立的 Proposal 與 Report，禁止跨階段混裝於單一巨石文件：

1. **Phase E1: 早期實體神經元定位（Localization & Amnesia Screening）**
2. **Phase E2: 下游注意力直接 Logit 歸因（Full-Attention DLA Attribution）**
3. **Phase E3: 上游抑制與下游衰減因果干預（Upstream Suppression & Downstream Attenuation）**

**收線報告：** [report-line-closing.md](report-line-closing.md)——整條線的敘事入口
（entity cell = 事實記憶載體而非決策單位；4 個確認 cell；9 條 shared fact
channel；V2 endpoint gate 被推翻、V3 fact gate 凍結）。

---

## 階段與版本矩陣（Phases & Version Matrix）

| 階段 / 版本 | 核心研究假說與方向來源 | 主要評估指標與合格門檻 | 對應專屬 CLI 命令 | 證據狀態（Evidence Status） |
|---|---|---|---|---|
| **[E1 V1](proposal-v1.md)** / [報告](report-v1.md) | 三行固定 Header 前綴變體 | Held-out 變體重疊 + 失憶門檻（雙關門檻） | `entity-cell run-localization --localization-family v1-header` | **Discovery 結案**：0/35 通過（31/35 撞車於模板神經元 L0 N4485；證偽固定 Header 定位有效性） |
| **[E1 V2](proposal-v2.md)** / [報告](report-v2.md) | 12 個自然句框（F0–F7 / H0–H3）消除固定前綴 | 四道門檻：held 重疊、form-robust、template-robust、amnesia 終點 | `entity-cell run-localization --localization-family v2-frames` | **Discovery 結案**：1/35 通過（FTNT, L0 N104；**證偽群體普遍性**，轉入單點解剖）。v2 儀器官方 GPU bf16 重驗（`entity-cell-e1-discovery-v4`）：四門檻全數通過，1/35 trusted 結論完全維持（見報告附錄） |
| **E1 V2 HFM（高事實記憶子集）** / [報告](report-hfm-discovery.md) | 同一 frozen E1 V2 protocol，6 家模型確實記得事實的高頻 entity（formal name surface forms） | 同 E1 V2 四道門檻 + factual amnesia probe（proposed） | 同上（prepared inputs 見報告） | **Discovery 結案**：1/6 通過（**AMZN, L0 N1476**，score 3386.7）。Factual amnesia probe：壓制 N1476 劑量依賴地崩塌 Amazon HQ 事實回想（-1.059 nats @ α=-3），wrong-cell / matched-random / 跨 ticker 對照全平穩。FTNT 的 factual recall preflight 為 NO-GO（模型無 FTNT 事實記憶可忘）。Decision-level probe（proposed）：壓制不產生 buy/sell 決策翻轉，決策層 entity prior 分散（見 [decision probe 報告](report-decision-probe.md)） |
| **[E1 V2 HFM-2（batch 2 跨 sector）](report-hfm2-discovery.md)** | 同一 protocol，20 家新候選篩選出 11 家（4 科技 + 7 跨 sector，scope label HighFactualMemory） | 同 E1 V2 四道門檻 + factual amnesia probe（proposed） | 同上（prepared inputs 見報告） | **Discovery 結案**：0/11 通過四門檻（9/11 被 endpoint gate 擋下，與決策層結論一致）。Fact-level probe 確認 **JNJ (L4, N7676)**（全線最強：HQ 崩塌 -8.51 nats、α=-3 下模型拒答 `___?`）與 **JPM (L0, N9025)**（三框特異、中等效應）；KO 部分特異；ORCL/UBER 否決。JNJ/JPM 完整 E2/E3 電池：0 決策翻轉、E3-B null、factual cell 與決策路徑解離（見 [battery 報告](report-jnj-jpm-battery.md)） | **Shared fact channel 發現**：13 個 fact probe targets 確認 **(L0, N4485) 一顆神經元承載 5 家（INTC/QCOM/Visa/WMT/XOM，另有 KO 部分）的 HQ/ticker 參數記憶**，對 AMZN 無效應；注意 N4485 是 V1 header 家族神經元，V2 frame localization 的 top-5 中只對 KO（top-1）/ORCL（#2）提名它（見 [E1 V3 proposal §9](proposal-v3.md)） |
| **E1 V3（frozen）** / [proposal-v3](proposal-v3.md) | 同一 localization（V2 不變）；Gate 4 從決策 margin 改為 **fact-level amnesia gate**（事實 cloze 崩塌 + 控制組 + gold 人驗）+ shared fact channel 分類（primary = 最高 rank 通過候選） | Gate 1–3 不變；Gate 4：≥1 frame collapse ≤ −0.5 nats 且 matched-random ≤ 0.3 nats 且 gold 人驗通過；跨實體 F0 檢查分類 entity cell vs shared fact channel | `entity-cell run-localization --stage e1-fact-amnesia` + `verify-fact-gold` | **Frozen**（calibration + hold-out 皆通過驗收，§10）：calibration **2 entity cell**（JNJ (L4, N7676)、PLTR (L2, N5003)）；hold-out（11 家新 entity，`entity-cell-e1-v3-holdout-v1`）**2 entity cell**（BAC (L0, N7801)、CAT (L2, N7997)）+ 2 新共享通道（L3N5456 LLY↔BAC、L2N5322 WFC↔BAC）+ (L4, N4184) 跨 batch 於 BAC 重現；控制組分佈與校準一致；gold 人驗通過率 55%（entity 依賴，fail-closed 正確） |
| **[E2](proposal-e2.md)** / [報告](report-e2.md) | 8 個 Full-Attention 層的 source-resolved DLA 歸因 | Identity vs Instruction 10:1 routing 標籤與加法重構誤差 | `entity-cell run-attribution --e2-layers 3 7 11 15 19 23 27 31` | **Discovery 結案**：128 heads 全部為 instruction-dominant；選出 5 個 heads 作為 E3-B 衰減組。v2 儀器重驗：top-5 選擇與 routing labels 全部不變（見報告附錄） |
| **[E3 V1](proposal-e3-v1.md)** / [報告](report-e3-v1.md) | 上游單元壓制 + 跨 Ticker 特異性對照 + 下游注意力路徑衰減 | 實體專屬性對比（FTNT vs 同撞車組 ADI/MU vs 異組 FTV） | `entity-cell run-intervention --peer-tickers ADI MU FTV` | **Discovery 完成**：實體專屬性成立。v2 儀器重驗：全部 frozen gates 通過；匿名基線在真決策下為 Buy（非 decision conflict）、0 flips（見報告 §6） |

---

## 階段關鍵科學發現摘要

1. **E1 定位階段總結**：
   大模型在 Qwen3.5-4B 規模下，不存在「每個公司擁有單一專屬 MLP 實體神經元」的普遍規律（V1 0/35 通過，V2 僅 1/35 通過）。自然句大幅瓦解了模板主導（N4485 佔比從 89% 降至 11%），但出現了新的實體槽位聚集（N104 為 17 家共享）。這明確表明：**4B 模型的公司實體記憶高度偏向分散式或高維編碼**。
2. **E2 歸因階段總結**：
   在財務提示詞的情境下，注意力機制對實體 Header 的關注度遠低於後續的證據與指令上下文（0/128 個 head 為 identity-dominant）。選定 L31 H0/1/3、L19 H4、L27 H6 作為下游干預（E3-B）的代表性 full-attention 路徑。
3. **E3 因果階段定位**：
   聚焦於對唯一通過四道門檻的單元 **`(L0, N104)`** 進行跨 Ticker 因果壓制，徹底定性其究竟為「FTNT 專屬實體單元」還是「17 家共享的通用實體槽位」。

---

## 歷史 Run 清單

所有 run 保存在 `artifacts/qwen3.5-4b/entity-cell-localization/runs/`：
- E1: `entity-cell-prepare-discovery-v1`, `entity-cell-e1-smoke-v2`, `entity-cell-e1-discovery-v1`
- E1 V2: `entity-cell-prepare-discovery-v2`, `entity-cell-e1-smoke-v4`, `entity-cell-e1-discovery-v2`, `entity-cell-e1-discovery-v3` (partial, CPU 自檢), `entity-cell-e1-discovery-v4` (complete, 官方 GPU bf16 重驗)
- E2: `entity-cell-e2-discovery-v1`~`v4` (failed, preserved), `entity-cell-e2-discovery-v5` (complete), `entity-cell-e2-discovery-v6` (attribution complete, v2 儀器)
- E3: `entity-cell-e3-discovery-v1` (timed out, preserved), `entity-cell-e3-discovery-v2` (failed downstream partition, preserved), `entity-cell-e3-discovery-v3` (complete), `entity-cell-e3-discovery-v4` (complete, v2 儀器重驗)
- E1 V2 HFM: `entity-cell-prepare-hfm-v2` (superseded, provenance), `entity-cell-prepare-hfm-v3` (complete), `entity-cell-e1-hfm-discovery-v1` (complete)；伴隨 compact outputs `factual_scan_broad.json`、`factual_recall_preflight.json`、`factual_amnesia_probe_amzn.json`（同 directory 根層，非 run dir）；decision-level probe outputs: `decision_flip_probe_amzn_neutral.json`、`suppression_nvda_n1786.json`、`suppression_amzn_positive.json`、`suppression_amzn_ladder.json`、`suppression_amzn_band.json` 與對應 `inputs_amzn_*.jsonl`（placeholder 證據，含 sha256 見 [decision probe 報告](report-decision-probe.md)）
- E1 V2 HFM-2: `entity-cell-prepare-hfm2-v1` (complete), `entity-cell-e1-hfm2-discovery-v1` (complete)；伴隨 compact outputs `factual_screen_batch2.json`、`factual_probe_{jnj,orcl,jpm,uber,ko}.json`（同 directory 根層）
- E2/E3 battery (JNJ/JPM): `entity-cell-e2-hfm2-v1` (complete, attribution + patching + analyze；readout 未執行—lens blocker 見 [battery 報告](report-jnj-jpm-battery.md))；伴隨 compact outputs `flip_probe_jnj_neutral.json`、`flip_probe_jpm_neutral.json`、`flip_probe_jnj_band.json`、`e3_records_jnj.json`、`inputs_jnj_band.jsonl`（placeholder 證據）
- Fact probe batch 2（shared slots + 剩餘 6 家候選）: `fact-probe-batch2/`（13 個 probe outputs + `targets.json`；proposed probe operator，[E1 V3 proposal §9](proposal-v3.md) 校準數據來源）
- E1 V3: `entity-cell-prepare-hfm2-v2` (complete, 含 fact_frames.jsonl), `entity-cell-e1-v3-hfm2-calibration-v1` (complete, calibration；gold 人驗 30/33), `entity-cell-prepare-holdout-v1` (complete), `entity-cell-e1-v3-holdout-v1` (complete, hold-out 11 家新 entity；gold 人驗 18/33；input 持久化於 `data/entity-cell/holdout-v1-*`)；V3 正式 run 含 `e1/fact_gold_verifications.json` 與 `analyze/summary.json` 的 `v3_candidate_eligibility`（見 [收線報告](report-line-closing.md)）

## 測量儀器註記（Instrument Note）

上述所有 Qwen3.5-4B run 的 margin / DLA 數值均產出於 shared core FP32 tail 的 v1 定義
（final norm 手動公式漏掉 Qwen3.5 的 `1+` 項，詳見
[`docs/shared-experiment-core.md`](../shared-experiment-core.md) 測量變更記錄 v2）。
結構性結論（同一 probe 內部的相對比較、no-flip 方向）維持自洽；絕對 margin 值與以 0 為界
的判定（含 FTNT P2/P3 的「匿名 Sell」前提，v2 儀器下為匿名 Buy）須以 v2 儀器重驗 run
為準。重驗狀態：

| 階段 | 重驗 run | 狀態 | 結果摘要 |
|---|---|---|---|
| E3 V1 | `entity-cell-e3-discovery-v4`（CPU fp32） | complete | 全部 frozen gates 通過（$A_p^{FTNT}=+0.0971$、ADI/MU 對照 $\Delta A_p$ +0.4104/+0.3932）；匿名基線在真決策下亦為 Buy（非 decision conflict）；324 records 0 flips；E3-B evidence preservation 100% 維持。見 [report-e3-v1 §6](report-e3-v1.md) |
| E1 V2 | `entity-cell-e1-discovery-v4`（官方 GPU bf16 完整重驗）+ `entity-cell-e1-discovery-v3` / 針對性重驗（CPU 輔助） | complete | 官方原生 GPU bf16 精度下四道門檻全過（FTNT form-robust 確認為 pass，overlap=0），1/35 trusted 結論完全成立；全 35 家 amnesia endpoint gate 共 13 家通過（含 v1 的 9 家與 4 家新通過），其餘 12 家皆被 form-robust/template 排除。見 [report-v2 附錄](report-v2.md) |
| E2 | `entity-cell-e2-discovery-v6`（CPU fp32，attribution 完整；analyze 階段因 latent import bug 失敗，`f885f20` 已修；summary 為 post-hoc 重算） | complete | top-5 head 選擇、排名與 routing labels 全部不變（DLA 放大约 1.4–1.5×）；見 [report-e2 §5](report-e2.md) |

---

## Proposed: factual recall preflight 與 factual amnesia probe（proposed，非 frozen protocol）

**動機。** E1 V2 的 amnesia endpoint gate 量測的是財務決策 margin 朝匿名基線的
移動，並未量測事實回想（factual recall）：即原定位方法（Barzilay et al. /
ROME 一系）驗證閉環的步驟 3——壓制候選單元後回到事實句子，度量模型是否遺忘
該實體的事實。目前 E3 V1 已確認 `(L0, N104)` 對 FTNT 財務決策 margin 的因果
特異性，但「該單元儲存 Fortinet 事實知識」仍無證據；且 preflight 顯示模型對
FTNT 的參數化事實記憶本身極弱（無事實可忘）。

**factual recall preflight（已執行）。** 建立 factual amnesia probe 之前的
go/no-go 檢查，純 prompt-driven：重用 E1 V2 frozen 的 factual-cloze frame
措辭（F0 headquarters / F2 ticker / F3 founded / H0 CEO），在 clean baseline
上捕捉模型對各 ticker 的完成式（top-5 distribution 與 greedy 續寫），並對照
Anonymous Company 表面控制。不向模型注入任何外部事實值；web-grounded 事實
僅用於人類側複核輸出。對原 E3 四家（FTNT/ADI/MU/FTV）結果為 **NO-GO**
（見 [report-hfm-discovery §6](report-hfm-discovery.md)）。

**factual amnesia probe（已執行，AMZN case）。** 對 E1 V2 HFM discovery 的
trusted candidate `(L0, N1476)` 壓制並量測 Amazon 事實回想崩塌：frozen gold
序列（clean greedy 前 3 token、人類側驗證）的 joint log-probability（teacher
forcing），6 點 dose curve、wrong-entity / matched-random 對照、跨 ticker
對照。結果見 [report-hfm-discovery §4](report-hfm-discovery.md)。

**decision-level suppression/flip probe（已執行）。** 對 buy/sell 決策層量測
frozen amnesia-endpoint margin（logP(buy)−logP(sell)，FP32 tail 儀器）：
方向掃描（entity vs anonymous header）、證據極性/強度搜尋（找到 1 個真正相反
組合：AMZN buy +1.18 vs anon sell -0.02）、dose sweep 翻轉測試、可選 fact
block、wrong-entity / matched-random / 跨 ticker 對照。結果：壓制 `(L0, N1476)`
不產生決策翻轉（只移除 entity prior 的 ~4%）；NVDA top-1 `(L2, N1786)` 壓制
不了 NVDA 的 entity 訊號。決策層 entity prior 是分散的，與 factual 層的特異性
崩塌並存——「entity cell = entity-specific factual representation，非決策驅動
單元」。結果與判讀見 [report-decision-probe](report-decision-probe.md)。

**Framing 註記。** Factual readout 使用 raw-text framing（無 chat template）：
chat template 下模型對短 factual cloze echo prompt，不可用於 factual recall
量測；localization 的 activation 記錄仍依 frozen protocol 用 chat template。

**Operator：** `scripts/entity_cell_factual_recall_preflight.py`、
`scripts/entity_cell_factual_amnesia_probe.py`、`scripts/entity_cell_fact_probe_batch.py`
（device 由 `CUDA_VISIBLE_DEVICES` 選擇；GPU 為 bfloat16）。前兩者為 proposed，
尚未凍結為 protocol；第三者為前者的批次 runner（共享 model load）。pre-registered
版本（gate 閾值、gold 人驗規則、shared fact channel 分類）見
[E1 V3 proposal](proposal-v3.md)（frozen；V3 run 為 entity cell 確認的
formal 判定）。


