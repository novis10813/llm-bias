# Entity Cell Localization and Downstream Attribution: Versions & Phases

本頁是 entity-cell 研究線（`llm_bias/entity_cell/`，CLI `entity-cell`）的文檔入口。依據 [`docs/AGENTS.md`](../AGENTS.md) 規範，本研究線解耦為三個獨立階段，每個階段各自維護獨立的 Proposal 與 Report，禁止跨階段混裝於單一巨石文件：

1. **Phase E1: 早期實體神經元定位（Localization & Amnesia Screening）**
2. **Phase E2: 下游注意力直接 Logit 歸因（Full-Attention DLA Attribution）**
3. **Phase E3: 上游抑制與下游衰減因果干預（Upstream Suppression & Downstream Attenuation）**

---

## 階段與版本矩陣（Phases & Version Matrix）

| 階段 / 版本 | 核心研究假說與方向來源 | 主要評估指標與合格門檻 | 對應專屬 CLI 命令 | 證據狀態（Evidence Status） |
|---|---|---|---|---|
| **[E1 V1](proposal-v1.md)** / [報告](report-v1.md) | 三行固定 Header 前綴變體 | Held-out 變體重疊 + 失憶門檻（雙關門檻） | `entity-cell run-localization --localization-family v1-header` | **Discovery 結案**：0/35 通過（31/35 撞車於模板神經元 L0 N4485；證偽固定 Header 定位有效性） |
| **[E1 V2](proposal-v2.md)** / [報告](report-v2.md) | 12 個自然句框（F0–F7 / H0–H3）消除固定前綴 | 四道門檻：held 重疊、form-robust、template-robust、amnesia 終點 | `entity-cell run-localization --localization-family v2-frames` | **Discovery 結案**：1/35 通過（FTNT, L0 N104；**證偽群體普遍性**，轉入單點解剖）。v2 儀器重驗：amnesia 門檻維持（見報告附錄；form-robust 為 bf16/fp32 near-tie） |
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
- E1 V2: `entity-cell-prepare-discovery-v2`, `entity-cell-e1-smoke-v4`, `entity-cell-e1-discovery-v2`, `entity-cell-e1-discovery-v3` (partial, v2 儀器自檢)
- E2: `entity-cell-e2-discovery-v1`~`v4` (failed, preserved), `entity-cell-e2-discovery-v5` (complete), `entity-cell-e2-discovery-v6` (attribution complete, v2 儀器)
- E3: `entity-cell-e3-discovery-v1` (timed out, preserved), `entity-cell-e3-discovery-v2` (failed downstream partition, preserved), `entity-cell-e3-discovery-v3` (complete), `entity-cell-e3-discovery-v4` (complete, v2 儀器重驗)

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
| E1 V2 | `entity-cell-e1-discovery-v3`（CPU fp32，partial preserved）+ 針對性 amnesia 重驗 | complete | FTNT amnesia 門檻在真決策下維持（2/3，與 v1 儀器相同模式）；endpoint-gate 集合 9→5 家（不影響 trusted 集合）；form-robust 為 bf16/fp32 精度 near-tie（官方 bf16 pass / fp32 fail，activation-based，與 norm 修復無關）；見 [report-v2 附錄](report-v2.md) |
| E2 | `entity-cell-e2-discovery-v6`（CPU fp32，attribution 完整；analyze 階段因 latent import bug 失敗，`f885f20` 已修；summary 為 post-hoc 重算） | complete | top-5 head 選擇、排名與 routing labels 全部不變（DLA 放大约 1.4–1.5×）；見 [report-e2 §5](report-e2.md) |
