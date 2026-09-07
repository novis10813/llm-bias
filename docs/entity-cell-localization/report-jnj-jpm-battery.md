# Entity Cell E2/E3 Battery for JNJ & JPM (proposed selection, frozen readouts)

**Status:** Complete。對 [HFM-2 discovery](report-hfm2-discovery.md) 確認的兩個
entity cell（JNJ `(L4, N7676)`、JPM `(L0, N9025)`）跑完整原本實驗組：
decision flip probe（中性 + 邊界帶）、E2 attribution、E3-A upstream + E3-B
downstream（JNJ，frozen record format）。**全部 0 個 buy/sell 決策翻轉**；
JNJ 邊界帶找到真正的相反組合（clean buy +0.09 / anon sell -1.21），壓制 JNJ
cell 不翻轉（margin 反而更 buy）。Selected late heads（E2 top-5）不承載 JNJ
決策 margin（E3-B null），L4 cell 經由這些 heads 的 mediation ≈ 0。

**Proposed selection 說明。** Frozen E3 CLI（`entity-cell run-intervention`）
從 E1 formal trusted eligibility 選 intervention target；HFM-2 discovery 是
0 trusted（9/11 被 endpoint gate 擋下），frozen CLI 無法驅動。本報告直接呼叫
frozen E3 record functions（`run_upstream_suppression_record` /
`run_downstream_suppression_record`）並明確指定 cell/heads，產出 byte-compatible
的 E3-A/E3-B compact records（含 flip 欄位）；selection 本身是 proposed
extension，記錄於 output provenance，不修改任何 frozen protocol 文件。

## Runs on record

`artifacts/qwen3.5-4b/entity-cell-localization/` 下：

| run / output | state | note |
|---|---|---|
| `flip_probe_jnj_neutral.json` | complete | JNJ/JPM/11 家中性財務 prompt 方向掃描 + JNJ `(L4, N7676)` flip sweep（`entity_cell_decision_flip_probe.py`） |
| `flip_probe_jpm_neutral.json` | complete | JPM `(L0, N9025)` flip sweep（reuse JNJ scan） |
| `flip_probe_jnj_band.json` + `inputs_jnj_band.jsonl` | complete | JNJ 邊界帶 8 prompts（placeholder 證據，重用 AMZN band 證據行 + JNJ header，identity_header ranges 取自 hfm2 prepared）：找到相反組合 |
| `runs/entity-cell-e2-hfm2-v1` | complete（attribution + patching + analyze） | E2 full-attention DLA，3 tickers（INTC/JNJ/JPM），8 layers；**e2-readout 階段未執行**（lens blocker，見下） |
| `runs/entity-cell-e2-hfm2-readout-v1` | complete（attribution + readout + analyze，2026-09-07） | 同一 hfm2 prepared 條件重跑 E2 並補上 e2-readout，在 pinned HF lens（`c2e20eb4…`）下執行、lens sha 明確 pin；head ranking 與 v1 完全一致，readout 結果見 §2 |
| `e3_records_jnj.json` | complete | E3-A（108 records）+ E3-B（60 records），JNJ 3 prompts × frozen dose grids × 5 selected heads（`entity_cell_e3_frozen_records_proposed.py`） |

**E2-readout lens blocker（pre-existing integrity gap）。** E2 readout 階段
的 lens 驗證要求 canonical lens metadata 的 `provenance.revision`；目前
on-disk lens（`artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt`，sha
`3691d7b2…`，local promote、chinese_simplified calibration）缺該欄位，且其
sha 與 `config/pretrained_lenses.json` pin 的外部 lens（`c2e20eb4…`，
Salesforce-wikitext calibration）不一致。E2 attribution / patching / analyze
不依賴 lens，正常完成；readout 需先依 [Qwen Jacobian-lens selection
proposal](../jacobian-lens-selection/proposal.md) 重新 pin/驗證 lens 後才可
執行。

**Blocker 已解決（2026-09-07）。** 依 repo 預設方向，用
`jacobian-lens install --replace-existing` 把 registry pin 的 HF pretrained
lens（`neuronpedia/jacobian-lens`，revision `a4114d77…`，wikitext calibration，
MIT）安裝為新 canonical（sha `c2e20eb4…`，metadata 含完整 HF provenance）；舊的
本地 chinese_simplified lens（`3691d7b2…`）自動 archive 至
`artifacts/qwen3.5-4b/jacobian-lens/archive/20260907T012716.057818Z/`（可還原）。
依 [Qwen Jacobian-lens selection proposal](../jacobian-lens-selection/proposal.md) 的規定，
兩種 lens provenance 不可無標示混用：本電池所有 readout 結果都在
**pretrained lens 條件**下產出，與本地雙語 lens 的讀數不可直接比較。

## 1. Decision flip tests（buy/sell 翻轉測試）

### JNJ 中性證據（3 prompts）

全數 sell/sell 同向（clean -1.59 ~ -2.70 vs anon -2.33 ~ -4.35），無相反
組合。壓 JNJ cell（α=-3）讓 margin 略偏 **buy**（方向**遠離** anonymous：
prompt 3 -1.591 → -0.598；wrong-entity / matched-random 平穩）——與 frozen
endpoint gate 預期方向（朝 anonymous）相反，是其 `endpoint_control_gate_below_2`
fail 的機制。0 翻轉。

### JNJ 邊界帶（8 prompts）

證據極性搜尋找到**真正的相反組合** `record_jbndfd3bf`（證據：「revenue grew
12 percent / margins held steady」）：

| | margin | greedy |
|---|---|---|
| clean JNJ | **+0.092** | **buy**（貼零） |
| clean anonymous | -1.213 | sell |

Flip sweep（target `(L4, N7676)`，all_positions）：

| α | +1.0 | 0.0 | -1.0 | -2.0 | -3.0 |
|---|---|---|---|---|---|
| margin | +0.092 | +0.195 | +0.238 | +0.254 | **+0.419** |

**不翻轉**——完整壓制讓 margin 更 buy（ap -0.251），wrong-entity（+0.131）與
matched-random（+0.061）同量級。JNJ 的 entity 決策偏置不由事實 cell 承載。

### JPM 中性證據（3 prompts）

全數 sell/sell 同向（gap -1.50 ~ -2.33）。壓 JPM cell（α=-3）margin 微幅偏
buy（ap -0.04 / -0.30 / -0.10；控制組平穩）。0 翻轉。

**小結**：4 個 entity（AMZN / NVDA / JNJ / JPM）× 3 種證據極性（中性 / 強
正面 / 邊界帶）× 找到 2 個真正的相反組合（AMZN、JNJ）——**entity cell 壓制
從未翻轉 buy/sell 決策**。

## 2. E2 attribution（`entity-cell-e2-hfm2-v1`）

3 tickers（INTC/JNJ/JPM）× 3 financial prompts × 8 full-attention layers
（L3/L7/L11/L15/L19/L23/L27/L31）的 DLA + 4 個 frozen donor conditions
（132/132 eligible）。Head ranking（frozen top-5 + sign consistency，
3/3 tickers consistency 1.0）：

| Head | Routing | identity mean | instruction mean |
|---|---|---|---|
| L31H0 | mixed | -0.0187 | -0.0585 |
| L31H1 | mixed | -0.0049 | -0.0428 |
| L31H3 | instruction-dominant | -0.0024 | +0.0604 |
| L27H7 | instruction-dominant | +0.0031 | +0.0507 |
| L27H6 | instruction-dominant | -0.0015 | +0.0308 |

與原 35 家 E2 的發現一致：**晚層（L27/L31）、instruction-dominant 為主**；
identity-header 貢獻普遍小。

### 2a. E2-readout 補跑（`entity-cell-e2-hfm2-readout-v1`，2026-09-07）

同一 prepared 輸入、3 tickers、8 layers，在 pinned HF lens 下重跑（lens sha
以 `--expected-lens-sha256` 明確 pin）。Attribution 確定性重現：top-5 heads
與 v1 完全一致。Readout 共 45 列：

- **27 列 unavailable（L31H0/H1/H3 × 3 tickers × 3 prompts）**：canonical
  lens 只覆蓋 L0–L30，L31 的 transported readout 依設計 fail-closed 記錄為
  unavailable（final block 另有 J = I 的讀法，不在本階段範圍）。
- **18 列 available（L27H6/H7 × 3 tickers × 3 prompts）**：top-k 為 transported
  head vector 經 FP32 final norm + LM head 的 logit，top-10 token 如下：
  - **L27H7（rank 4）**：三家 top-10 幾乎相同——梯度/gradient、刮/划伤/Scar、
    迁移/migration、孵化器/incubator、architect 一組 entity-無關的抽象技術語義
    簇；同一 head vector 的 buy−sell logit margin 在全部 9 個條件上穩定為
    **+2.26 ~ +2.91**（公司無關的輕度 buy 傾斜）。
  - **L27H6（rank 5）**：碎片 token 簇（pers/Pers、Simpson、夹、拒、pin、ross、
    clip），同樣跨 ticker 不變；buy−sell margin ≈ ±0.2，無決策信號。
- **沒有 entity 名稱 token**（INTC/JNJ/JPM）出現在任何 top-10——與本線核心發現
  一致：entity 內容在 MLP fact cells，不在 top-ranked attention heads 的
  transported readout 裡。

解讀限制（readout 記錄內的 interpretation_limit）：這是 transported
representation readout，token ranks 與 scores 都是描述性的，不建立 attention、
chain-of-thought、離散 reasoning path 或因果證據。

## 3. E3-A upstream（JNJ cell `(L4, N7676)`，frozen record format）

3 prompts × 2 scopes × 6 alphas × 3 controls（target / wrong-entity
`(0, 1476)` / matched-random `(L4, N9197)`）= 108 records。

- **0 翻轉**。Margin 在 3 個 prompt 上皆朝 buy 方向微移（同 flip probe）。
- **Mediation**（cell 壓制對 5 個 selected heads DLA 的傳播，α=-3）：
  max |Δ| ≈ 0.007，sum|Δ| = 0.042（wrong-entity 0.0085、matched-random
  0.0088）——L4 cell 幾乎不經由 selected late heads 路由到決策 margin。

## 4. E3-B downstream（5 個 selected heads，frozen record format）

3 prompts × 5 betas（1.0 → 0.0）× 4 modes（identity / evidence /
random_subset / whole_head）= 60 records。

- **完整 null**：全部 attenuate（β=0，identity source）也只移動 ±0.03 nats
  （noise 範圍），0 翻轉。Selected late heads 不承載 JNJ 的 buy/sell 決策
  margin。

## 5. 判讀

1. **Factual cell 與決策路徑解離**：JNJ `(L4, N7676)` 在事實層有整條線最強
   的因果效應（HQ -8.51 nats、拒答行為、控制組平穩），但在決策層既不驅動
   margin 方向（壓制偏 buy 而非 anon），也不經由 top-5 late heads 傳播，
   且 late heads 本身不承載 margin。Entity 事實記憶與 buy/sell 決策是
   **兩條不同路徑**。
2. **Endpoint gate 的方向假設不成立**：frozen gate 預期壓制 entity cell 會
   讓決策朝 anonymous 移動；對 JNJ/JPM（及 AMZN 的正向 band）實際方向是
   反的（略偏 buy）。Gate 對這些 cell 的 fail 不是「效應太小」，而是
   「方向不符合假設」——決策層 entity prior 的機制不是 cell 承載。
3. **決策 flip 在單單元壓制下不可達**（跨 4 entity、3 證據極性、2 個真正
   相反組合）：與 [decision probe 報告](report-decision-probe.md) 的結論
   一致並強化。

## 6. 限制

- E3 selection 為 proposed（formal eligibility 未通過）；records 是 frozen
  format，但「哪個 cell/head 被選」不是 frozen gate 的產物。
- E2 只跑 3 tickers（INTC/JNJ/JPM）；head ranking 的跨 ticker 一致性在 3
  家上評估。
- E2-readout 在 pinned HF lens（wikitext calibration）條件下執行；L31 的
  top-3 selected heads 無法 readout（canonical lens 覆蓋 L0–L30）。Readout 與
  本地雙語 lens 是不同實驗條件，不可混讀。
- JPM 未跑 E3（事實層效應中等；JNJ 為本電池主標靶）。

## 7. Next steps（建議）

1. **凍結 fact-level amnesia 門檻**（JNJ/AMZN/JPM 效應量 + 控制組分佈為
   依據），讓 entity cell 確認成為 frozen 判定；屆時 frozen E3 CLI 可正式
   驅動（JNJ 應為 trusted）。
2. ~~重 pin/驗證 canonical lens（解 E2-readout blocker）。~~ 已完成
   （2026-09-07：安裝 pinned HF lens，readout 補跑於
   `entity-cell-e2-hfm2-readout-v1`）。
3. 若要決策層 claim：需 population-level 多單元壓制或 head-level
   intervention 的系統搜尋（本電池的 E3-B 已排除 top-5 late heads）。
