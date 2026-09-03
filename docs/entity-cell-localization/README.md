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
| **[E1 V2](proposal-v2.md)** / [報告](report-v2.md) | 12 個自然句框（F0–F7 / H0–H3）消除固定前綴 | 四道門檻：held 重疊、form-robust、template-robust、amnesia 終點 | `entity-cell run-localization --localization-family v2-frames` | **Discovery 結案**：1/35 通過（FTNT, L0 N104；**證偽群體普遍性**，轉入單點解剖） |
| **[E2](proposal-e2.md)** / [報告](report-e2.md) | 8 個 Full-Attention 層的 source-resolved DLA 歸因 | Identity vs Instruction 10:1 routing 標籤與加法重構誤差 | `entity-cell run-attribution --e2-layers 3 7 11 15 19 23 27 31` | **Discovery 結案**：128 heads 全部為 instruction-dominant；選出 5 個 heads 作為 E3-B 衰減組 |
| **[E3 V1](proposal-e3-v1.md)** | 上游單元壓制 + 跨 Ticker 特異性對照 + 下游注意力路徑衰減 | 實體專屬性對比（FTNT vs 同撞車組 ADI/MU vs 異組 FTV） | `entity-cell run-intervention --peer-tickers ADI MU FTV` | 協議凍結；實作修復中，準備執行 Preflight 與 Discovery run |

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
- E1 V2: `entity-cell-prepare-discovery-v2`, `entity-cell-e1-smoke-v4`, `entity-cell-e1-discovery-v2`
- E2: `entity-cell-e2-discovery-v1`~`v4` (failed, preserved), `entity-cell-e2-discovery-v5` (complete)
- E3: `entity-cell-e3-discovery-v1` (timed out, preserved), `entity-cell-e3-discovery-v2` (failed downstream partition, preserved)
