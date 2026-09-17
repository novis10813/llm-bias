# Evidence-insensitivity — Phase 1 行為篩選報告

**狀態**：Phase 1 completed（2026-09-16）。G-P1..G-P4 全數通過、fallback 未觸發、
503 家分組表 frozen。  
**對象模型**：Qwen3.5-4B（bf16）。  
**協議**：[proposal（frozen Rev 2）](proposal-phase1.md)。

---

## 執行概覽

- **Run ID**：`phase1-gpu-bf16-01`
- **Run root**：
  `artifacts/qwen3.5-4b/evidence-insensitivity/runs/phase1-gpu-bf16-01`
- **規模**：503 家 × 9 條件（primary）＋ 100 家 order-swap × 8 條件 ＋ anonymous
  9 = **5336 forwards** ＋ 20 筆 determinism re-run
- **Pilot（pre-flight）**：`pilot-20260916T090440Z`（259 forwards；技術門檻全過，
  見協議 Rev 2 紀錄）
- **Operator**：`scripts/evidence_insensitivity_phase1.py`（`prepare` / `forward` /
  `analyze` stage subcommands）
- **Manifest**：`complete`，3/3 stage complete，5 artifact 登記
- **執行註記**：首次 forward 因外部 session 斷線被 kill（死於模型載入後、未寫任何
  record）；依 stage 邊界以 `forward` subcommand 續跑，全部 5336 筆 record 來自
  單一過程、determinism re-run bit-exact，provenance 不受影響

## Gates：四項全過，fallback 未觸發

| Gate | 門檻 | 觀測值 | 通過 |
|---|---|---|:---:|
| G-P1 parse | ≥ 0.95 | **0.9998**（5336 中 1 筆 fail：VICI N10，reason 過長於 128 token 截斷、JSON 未閉合） | ✓ |
| G-P2 符號一致 | ≥ 0.90 | **0.9738** | ✓ |
| G-P3 determinism | 0 mismatch / ΔM=0 | **0 / 0.0**（20 筆 re-run bit-exact） | ✓ |
| G-P4 分組功效 | ≥ 10 且 ≥ 10 | **responsive 50 / insensitive 453** | ✓ |

生成格式：5336/5336 首 key 為 decision（pretty-printed JSON、尾端 stop marker，
parse 以 `raw_decode` 提取，見協議 §2）。

## 核心結果

### 1. 零證據 sell prior 為全人口普遍現象

- **503/503 家零證據決策 sell**（d0 分佈全 sell）；mean M(zero) **−5.40**
  nats（IQR 0.91）。
- Anonymous（身分剝除）zero M=**−6.75**：匿名比具名更偏 sell，但差距遠小於
  證據梯級效應（~4.9 nats 跨 zero→P 條件）。
- prior-reversed 組為 0（所有 fixed-sell 公司零證據皆 sell，prior-consistent
  453/453）。

### 2. Frozen 分組表（503 家）

| Group | 家數 | 比例 | discovery | hold-out |
|---|---:|---:|---:|---:|
| evidence-responsive | **50** | 9.9% | 42 | 8 |
| fixed-sell | **453** | 90.1% | 360 | 93 |
| fixed-buy | 0 | 0% | 0 | 0 |
| mixed | 0 | 0% | 0 | 0 |

分組規則（協議 §5）：responsive = D(N15)=sell 且 D(P15)=buy。分組表完整值在
`analyze/summary.json`。50 家 responsive：

AAPL、ACN、ADP、ADSK、ALLE、AMD、AMT、AMZN、APD、AVGO、BRK-B、CAT、CDNS、
CI、CME、CRM、CRWD、DIS、EQIX、EXR、GD、GOOG、GOOGL、GRMN、ICE、INTU、JPM、
LIN、LLY、LOW、MA、MCD、MPWR、MRK、MSFT、MU、NEE、NVDA、ORLY、PANW、PLD、
PWR、SLB、SNPS、TMO、TMUS、TXN、UNH、V、VRTX

sector 結構（responsive / sector 總數）：Information Technology **16/65
（24.6%）**、Financials 6/71、Health Care 6/58、Consumer Discretionary 5/44、
Industrials 5/77、Communication Services 4/19、Real Estate 4/31、Materials 2/24、
Energy 1/21、Utilities 1/31、**Consumer Staples 0/33、Unspecified 0/29**。
IT 顯著過代表（24.6% vs 人口基率 9.9%）；完整 sector 表在 summary。

### 3. Margin 對每家公司都跟隨極性，但決策閾值被 sell prior 壓死

- **503/503 家 response contrast C_c = M(P15)−M(N15) > 0**（範圍 0.66–2.04
  nats）：證據極性在 logit 層對全人口有效。
- population 極性曲線（mean M）：zero −5.40 → N15 −1.80 → N10 −1.42 → N8
  −1.23 → N6 −0.90 → P6 −0.50 → P8 −0.53 → P10 −0.65 → P15 −0.54。
  跨證據條件的移動（~4.9 nats）遠大於公司間零證據差異（IQR 0.91）。
- 決策層 buy rate：zero 0%、N15 0%、N6 1.6%、P15 9.9%（50 家，即
  responsive 組）、P6 11.1%（56 家）。**buy rate 對 P 條件非單調**（P6 > P15，
  描述性；與 pilot 同方向，小差異來自閾值邊公司的離散跳躍）。
- 解讀：離散決策的 sell 閾值位於 M ≈ 0 之下（pilot 的 DLR/ORCL 型：M 已正仍
  sell）；90.1% 的公司在 POS-first 順序下推不過閾值。

### 4. Order-swap 臂：項目位置是強、非對稱的決策驅動（recency）

約定：primary 臂 POS 恆為 item1；order-swap 臂兩項對調（NEG 先、POS 後）。

- **800 對（100 家 × 8 條件）中 366 對決策翻轉，全部 sell→buy**。
- 按條件（每條件 100 家）：**P6 69、P8 90、P10 88、P15 89** vs N6 14、N8
  6、N10 8、N15 2。
- mean |ΔM|：P8/P10/P15 ≈ **1.30–1.35**、P6 0.71 vs N6–N15 ≈ 0.31–0.43。
- 解讀：P 條件 swap 後**主導（正）證據移到最後**→ 大量 sell→buy；N 條件 swap
  後主導（負）證據移到最后 → 幾乎不翻。即**決策跟隨最後一項證據**，且非對稱：
  正項在尾強推 buy、負項在尾不推 buy（sell 為 attractor）。此與
  entity-to-dial 收線幾何的「sell attractor 待確認」方向一致。
- 量級：order 效應（|ΔM| ≈ 1.3，P 條件）與極性效應（C_c ≈ 1.3）在決策層同量
  級。分組表定義於 primary 順序（POS-first）；fixed-sell 標籤的語義是「POS
  在前時不跟隨證據」。order 作為顯式實驗條件不在 V1 範圍（版本分立觸發
  條件 #1），留作未來版本候選。

### 5. Reason 文本簽章（descriptive）

- 零證據 sell：「No evidence provided to support a buy decision」——模板層級的
  證據缺失規則，非公司特定先驗語言。
- fixed-sell @ P15（淨正證據）固定話術：「the **immediate** risk of margin
  compression … **outweighs** the (speculative) catalyst, **net negative**
  outlook」。
- responsive @ P15 翻轉話術：「catalyst … **outweighs the temporary** margin
  compression risks」。
- 同一對證據、同一模板下，兩組差異是**對證據的框架**（負=立即且量化 vs
  正=投機且暫時）；此為 Phase 2 L15 狀態對比要捕捉的現象。

## 邊界與限制

- 行為端點；無介入、無 causal claim（協議 §11）。
- 單模型（Qwen3.5-4B）、greedy、英文、受測模板。
- 分組表為 Phase 2/3 輸入；機制相只可用 discovery split（42 responsive / 360
  fixed-sell）選座標，hold-out（8 / 93）留 confirmation。
- 1 筆 parse fail（VICI N10 截斷）：該公司 N15/P15 皆 parse 成功、分組不受
  影響。

## 下一步

Phase 2 協議起草（L15 組間對比）：discovery split 42 responsive vs 360
fixed-sell；stance 軸於新 prompt family 重新 derive 並對 margin 正交化；sector
控制。描述統計設計輸入（2026-09-16 用戶討論，已記入線入口）：per-company 狀態
反應曲線（stance 投影 × polarity 梯級）、offset vs gain 分解（閾值故事 vs
承載衰減故事）、行為組 × P15 狀態方向 2×2 交叉表。另：order 效應（§4）若列為
顯式條件需另立版本。
