# Balanced-evidence-gap 研究線收線報告

**狀態**：收線（2026-09-10）  
**對象模型**：Qwen3.5-4B（bf16，hybrid Gated DeltaNet 架構）  
**核心結論**：實體身份在平衡財務證據下能誘發具統計穩定性的決策落差（Phase 1 與 2A 全數通過），且其內部表徵在殘差流中具備清晰的層級傳遞特徵（Phase 2B 定位至 L0–11 承載、L12–15 交接）。然而，由一階敏感度歸因篩選出的三個實體候選 MLP 神經元（L19、L20、L26），在加性因果干預下無一能產生超越隨機對照組的決策偏轉（Phase 3 驗證為 0/3 confirmed）。本研究線證實實體落差為真實存在的行為現象，但一階歸因篩選出的單神經元並非其因果傳遞管道。

**名詞定義**：

- **balanced-evidence gap**：在多空財務論據嚴格對稱之模板下，模型單純因置換實體名稱所產生的 logit margin 偏移量。
- **logit margin（$M$）**：模型在固定格式輸出中預測 buy 與 sell 標記的未正規化 logit 差值（$\text{logit}(\text{buy}) - \text{logit}(\text{sell})$）。
- **clean_margin**：無任何內部狀態置換或神經元偏移條件下的原始決策 margin。
- **mlp_addition**：對特定層與特定神經元的 down-projection 激活值，於所有序列位置施加常數偏移量 $\delta$ 之干預手段。

## 1. 實體誘發的決策落差在行為層確認成立，且脫離特定證據後仍維持跨情境穩定

在多空論據數量與強度對稱之受控條件下，實體置換能獨立誘發不可忽視的投資決策落差：

- **Phase 1 行為基準確認（16 家企業 × 4 大產業）**：在固定財務證據的前提下，單純置換企業名稱即可導致決策 margin 產生跨度達 1.5 nats 的系統性偏移。三項預先凍結之檢驗判準全數通過，證據順序重排後的落差排序相關係數達 $\rho = 0.941$，證實該落差並非輸入文本順序所致之偶發雜訊。
- **Phase 2A 跨情境穩定性檢驗（4 種語法變體）**：當財務證據改以四種中性論述模板呈現時，純實體造成的 margin 排序與 Phase 1 之落差維持顯著正相關（Rev 2 評估 $\rho = 0.729 > 0.3$，群組構念檢驗 4/4 通過）。該結果確立了實體先驗在跨語法環境下的客觀存在，排除了特定句型誘發假象的可能。

## 2. 實體表徵於中間層存在明確傳遞帶，並在指令上下文讀出層達到轉移峰值

透過對殘差流進行跨層激活置換掃描（activation patching layer sweep），實體資訊在模型內部的傳遞呈現結構化分工：

- **底層實體特徵承載帶（L0–11）**：實體 token 區間的狀態置換在 L0 至 L5 展現近乎完全的決策轉移力（正規化轉移率 $T \approx 1.0$），並一路持續至 L11；此後該區間的轉移效果急遽衰減，至 L12 之後 95% 信賴區間已包含 0。
- **特徵移轉與指令讀出峰值（L12–15）**：在 L12 至 L15 的過渡窗口中，決策轉移能力由實體區間交接至指令上下文區間。指令區間之轉移率於 L15 達到全局峰值（$T = +0.464$，95% 信賴區間 $[+0.37, +0.55]$），確立了高層決策形成前的關鍵轉譯位置。

## 3. 一階歸因篩選出三個具統計顯著之候選通道，但注意力頭未見特異性訊號

針對晚期決策形成區間進行組件歸因（component attribution），模型在 MLP 與注意力機制間表現出極端不對稱：

- **注意力頭呈全域虛無（80 個全注意力頭）**：在包含 L15、L19、L23、L27、L31 等關鍵層的 80 個注意力頭中，實體歸因與 margin 之相關係數最大值僅為 $\rho = +0.0141$（Holm 校正後 $p = 1.0$），未見任何具備實體特異性的注意力結構。
- **MLP 通道檢出三處強相關坐標**：在全域多重檢定校正下，三個 MLP down-projection 激活通道勝過同層隨機抽樣對照組，且在四大產業內部均呈現完全一致之符號方向（產業一致性 4/4，Holm 校正後 $p = 0.0418$）：
  1. **L19 / n6334**：$\rho = -0.897$（對照組最大值 $0.618$）；
  2. **L20 / n6520**：$\rho = +0.894$（對照組最大值 $0.635$）；
  3. **L26 / n2394**：$\rho = -0.859$（對照組最大值 $0.538$）。

## 4. 加性因果介入全數落入擾動底噪，證實一階相關通道不具備決策操縱力

對 Phase 2C 檢出之三處神經元實施加性偏移干預（$\delta \in \{0, \pm s, \pm 2s, \pm 4s\}$，其中 $s$ 為激活絕對值之 90 分位數），因果檢驗結果全數為偽（Gate 3A certified = 0/3）：

- **效應量級未能超越對照組**：在驗證點 $\pm 4s$ 強度下，三個候選通道誘發的平均 margin 位移極其微弱（$|\text{mean } \Delta M| \le 0.012$ nats），全數低於同層隨機挑選之 10 個對照神經元所產生的平均最大位移（對照組底噪為 $0.025$ 至 $0.051$ nats）。
- **決策翻轉數為零**：16 家企業在所有介入強度下均維持原始之 sell 判定，無任何樣本發生多空翻轉。
- **全域對照組彰顯通道微弱性**：在相同介入強度下，已知之 investment dial 坐標（L15 / n8490）能推動 margin 產生 $\pm 1.0$ 至 $1.1$ nats 的大幅位移。此對照證實干預機制本身有效，三個候選實體通道的效應微弱並非度量工具失靈所致。

| 候選坐標 | 預測方向 | 門檻 $\delta$ | 平均 $\Delta M$ | 符號一致比例 | 校正後 $p$ 值 | 對照組最大位移 | Gate 3A 判定 |
|---|---|---|---|---|---|---|---|
| L19 / n6334 | sell | $-0.4785$ | $+0.0122$ | 7 / 16 | $1.0000$ | $0.0334$ | **FAIL** |
| L20 / n6520 | buy | $+0.7305$ | $+0.0015$ | 8 / 16 | $1.0000$ | $0.0514$ | **FAIL** |
| L26 / n2394 | sell | $-0.7812$ | $-0.0056$ | 11 / 16 | $0.3152$ | $0.0252$ | **FAIL** |

## 5. 一階歸因僅屬關聯探索指標，其相關符號不可直接推論為因果極性

Phase 3 實測響應揭示了一階歸因在因果推論上的方法論局限：

- **極性翻轉現象**：L26 通道在介入後呈現清晰且單調的劑量響應，但其作用方向為正向推升 margin（$\delta > 0 \to \text{buy}$），與一階歸因相關係數所隱含之空頭預測（$\rho = -0.859$）完全相反。
- **無方向通道**：L20 通道在正負干預下均未展現單調趨勢，其微小位移散落於隨機震盪區間，與 pilot 階段測得之局部導數（$\partial M / \partial a \approx -0.0018$）一致。
- **非因果保證**：一階歸因本質上量測「該通道之激活敏感度在不同樣本間的分佈特徵」，其相關係數僅反映敏感度數值與樣本原始立場的共變關係，並不保證人為施加激活偏移時模型輸出的流向。此一現象與 financial-soundness 研究線之虛無結果相互呼應，表明一階歸因不應作為獨立的因果定位證據。

## 6. 研究宣稱之邊界與未涵蓋事項

- **干預機制採用全位置廣播（all-position addition）**：為與 investment dial 技術標準保持一致，本實驗對所有序列位置施加相同之 $\delta$。若實體通道之因果效應高度局限於特定 token 位置，全位置廣播可能帶來稀釋效應；本線不排除局部序列遮罩干預之潛在價值，但該方法須另立版本規範。
- **樣本規模局限於 16 家企業**：所有實驗均在投資立場測試集（`test` split）之 16 家具備完整配對的企業樣本上進行，未向外擴展至 427 家全域企業庫。
- **模型與語言專一性**：所有觀測均基於 Qwen3.5-4B 與英文財務論據模板，不主張此結論能直接外推至不同規模之模型或多語言場景。

## 7. 收線判定：核心現象確認成立，因果收口宣告虛無

本研究線之核心任務「實體身份如何誘發決策落差及其神經傳遞路徑」已取得階段性解答：
1. **現象層**確認實體落差具備顯著行為強度與結構穩定性；
2. **表徵層**確認實體資訊自底層殘差流移轉至中高層指令區間之幾何路徑；
3. **組件層**證實基於一階歸因挑選的晚期 MLP 單神經元無法承載該落差之因果推力。

本線既定之因果驗證假說已被拒絕，既有協議已全數執行完畢，本研究線正式收線。後續若有新探索方向，應開立獨立研究線：
- 針對實體 token 區間施加序列遮罩干預之局部因果檢驗；
- 針對 L0–11 殘差流承載帶之表徵幾何分解與多通道聯合干預；
- 跨模型架構對實體決策偏誤之橫向比對。

## 產物與數據索引

- **Phase 1 行為基準**：
  - 協議文檔：`docs/balanced-evidence-gap/details/proposal-phase1.md`
  - 執行記錄：`artifacts/qwen3.5-4b/balanced-evidence-gap/runs/balanced-gap-gpu-bf16-01`
  - 分析報告：`docs/balanced-evidence-gap/details/report-phase1.md`
  - 核心圖表：`docs/assets/balanced-evidence-gap/balanced_evidence_gap.{pdf,png}`
- **Phase 2 狀態與組件定位**：
  - 協議文檔：`docs/balanced-evidence-gap/details/proposal-phase2.md`（Rev 1）與 `details/proposal-phase2-rev2.md`（Rev 2）
  - 執行記錄：2A `phase2a-gpu-bf16-01`、2B `phase2b-gpu-bf16-01`、2C `phase2c-gpu-bf16-05`、2C 重評 `phase2c-gate-reanalysis-01`
  - 分析報告：`docs/balanced-evidence-gap/details/report-phase2.md`
  - 核心圖表：`docs/assets/balanced-evidence-gap/phase2b_layer_sweep.{pdf,png}`、`phase2c_components.{pdf,png}`
- **Phase 3 神經元因果驗證**：
  - 協議文檔：`docs/balanced-evidence-gap/details/proposal-phase3.md`（Rev 2 frozen）
  - 執行記錄：`artifacts/qwen3.5-4b/balanced-evidence-gap-phase3/runs/phase3-gpu-bf16-02`
  - 分析報告：`docs/balanced-evidence-gap/details/report-phase3.md`
  - 核心圖表：`docs/assets/balanced-evidence-gap/phase3_neuron_causal.{pdf,png}`
- **程式與測試模組**：
  - 核心套件：`llm_bias/balanced_evidence_gap/`（`template.py`, `spans.py`, `analysis.py`, `intervention.py`, `patch_pipeline.py`, `rev2.py`, `gate_reanalysis.py`, `neuron_causal.py`）
  - 執行腳本：`scripts/balanced_evidence_gap*.py`
  - 回歸測試：48 項測試全數通過（`tests/test_balanced_evidence_gap_*.py`）
