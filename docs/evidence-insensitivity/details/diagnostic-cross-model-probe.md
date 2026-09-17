# Cross-model diagnostic：sell 預設是 prompt 特性還是模型特性？（2026-09-16）

**定位**：Phase 1 收線後的輔助診斷（非協議 run）。目的：裁決 Phase 1 觀測到的
極端 sell 預設（503/503 zero-evidence sell、M −5.4 nats）主要是 prompt 模板
特性還是 Qwen 模型特性。

**方法**：同一支 frozen Phase 1 模板＋共用證據 family，在同一 50 家
pilot 公司（seed 20260916）上換模型重跑 pilot 規模（50 × 9 條件＋9 匿名
= 259 forwards）。

- **Run ID**：`cross-model-llama32-1b-01`（Llama-3.2-1B-Instruct，Meta 家族，
  dense 1B，bf16）、`cross-model-llama32-3b-01`（Llama-3.2-3B-Instruct，同家族
  3B，unsloth mirror 下載，bf16；1B→3B 作 size 控制）、`cross-model-gemma4-e2b-01`
  （Gemma-4-E2B-it，Google 家族，MatFormer E2B，bf16；HF 直下＋官方
  `chat_template.jinja` 注入 tokenizer_config）
- **Run root**：`artifacts/llama-3.2-1b-instruct/evidence-insensitivity/runs/cross-model-llama32-1b-01`、
  `artifacts/llama-3.2-3b-instruct/evidence-insensitivity/runs/cross-model-llama32-3b-01`、
  `artifacts/gemma4-e2b-it/evidence-insensitivity/runs/cross-model-gemma4-e2b-01`
- **對照 run**：`pilot-20260916T090440Z`（Qwen3.5-4B，同 50 家同條件）
- **實作**：Phase 1 pipeline 增加 `model_slug` 參數（artifact root 命名參數化；
  預設行為與 frozen 語義不變）。Llama tokenizer 的 buy/sell 為多 token，margin
  走 core continuation fallback（scoring_mode 已記錄）。

## 結果對照（同 50 家、同條件）

| 維度 | Qwen3.5-4B | Llama-3.2-1B | Llama-3.2-3B | Gemma-4-E2B | 歸屬 |
|---|---|---|---|---|---|
| 零證據行為 | 50/50 sell，M −5.29 | **50/50 安全拒答**（「I can't provide personalized financial… advice」，甚至全部幻覺成 blockchains），無 buy/sell，M ≈ −0.10 | 50/50 sell，M −1.08（無拒答） | 50/50 sell，M **−7.01**（四模型最強 sell） | 模板可靠地引出保守輸出（四模型零證據皆不買）；強度與形式是模型特性 |
| 極性梯度（margin） | −5.29 → −0.45（跨 4.9 nats，單調） | **≈ −1.4 平坦**（N15 −1.40 → P15 −1.36，0.04 nats） | N15 −5.87 → P15 −4.89（**~1.0 nats**；主要移動是「有無證據」：zero −1.08 → 任何證據 ≈ −5） | 決策層：zero sell → **N6/P6/P15 全 100% buy、N15 18% buy**（「有任何證據就 buy」模式，非極性跟隨；margin 在代碼圍欄前綴位置與決策分歧，G-P2 0.55） | 證據反應結構完全是模型特性（同模板三種不同結構） |
| entity 分化（named − anon） | +0.36～+1.35 nats；50/503 responsive | **≈ +0.04 nats；0/50 responsive、50/50 fixed-sell** | 0/50 responsive、49 fixed-sell；buy 決策 0/450 具名 prompt | +1.19 nats（zero）；41 responsive＋9 fixed-buy、0 fixed-sell（responsive 不集中知名股：PG/UNP/VZ/KDP 皆在） | entity 效應 Qwen 有、Gemma 有（方向不同）、Llama 無 |
| 有證據時的決策 | 90% sell、10% 跟隨 | **parse 成功者 100% sell**，reason 固定引用負證據（「execution timelines and… margin compression」） | 100% sell（parse 0.99） | **~90% buy**（除最強負 N15 外） | 保守 sell 預設只在「零證據」層是共通的；有證據後的結構各異 |
| 格式遵循 | JSON 直出，parse 1.0 | JSON 包在 ` ```json ` 代碼圍欄內，zero 條件全數拒答（parse 0.80） | 多數 JSON 直出、部分代碼圍欄（parse 0.99） | JSON 包代碼圍欄，parse 1.0（raw_decode 可解析） | 1B 指令遵循最弱 |

## 解讀

「極強 sell」拆成兩層：

1. **Prompt 層（跨模型可靠）**：模板的舉證框架（buy 需要證據支持、無 hold
   選項、「make a final investment decision」）對兩個家族都產生保守輸出——
   Qwen 以「no evidence to support a buy」逻辑 sell，Llama 以安全拒答或
   固定引用風險 sell。模板不產生 buy 傾向。
2. **模型層（本線研究的對象）**：(a) Qwen 的零證據強 sell stance（−5.3
   nats；Llama-1B 只有拒答，Llama-3B 的強 sell 在「有證據」條件才出現，
   Gemma 零證據 sell 更強 −7.0）；(b) 4.9 nats 的**分級極性跟隨**是
   Qwen 結構（Llama-1B 平坦 0.04、Llama-3B 約 1.0 nats 且動能是證據存在、
   Gemma 是「有任何證據就 buy」的非極性模式）；(c) entity 世界知識對公司間
   結構的分化：Qwen 有（50 家 responsive 幾乎全高知名度成長名）、Gemma
   有但方向不同（41 responsive＋9 fixed-buy、不集中知名股）、Llama 1B/3B
   完全同質（0/50 responsive）。

**結論**：同一支模板在三個家族上產生**三種不同的證據反應結構**（Qwen：
分級極性跟隨＋10% entity 分化；Llama：全面保守 sell；Gemma：零證據 sell
但有任何證據就 buy）。零證據的保守 sell 預設是模板可靠引出的共通底層
（prompt 層）；「證據不靈敏度作為 entity 分化的行為特質」及其具體形式
（誰跟隨、跟隨多少、sell 還是 buy 固定）是**模型特性**，不是模板假象。
這同時強化了 Phase 2 的動機：453 vs 50 的 L15 狀態差量測的是 Qwen 內
真實的模型機制，不是 prompt artifact。

## 限制

- Meta 家族兩檔 size（1B/3B）顯示 size 效應存在（拒答消失、stance 變強），
  但兩檔皆無分級極性反應與 entity 分化——Qwen-4B 相對於 Llama-3B 的 size
  差距不足以解釋該結構差異。
- Gemma-4-E2B 為 MatFormer「effective 2B」（raw ~5B）；其「有任何證據就
  buy」模式與 1B/3B Llama 的保守模式顯示同家族不同 size 也可能不同結構，
  家族×size 交互未完全分離。
- Gemma 的 margin 在固定 canonical 前綴位置量，但其決策 token 在代碼圍欄
  內（G-P2 0.55）——margin 曲線解讀需謹慎，**決策**是主要訊號。
- gemma4-12b/E4B（16GB+）因 GPU 共享空間不足未跑；gemma4-31b-it bf16
  59GB 單卡裝不下。
- Llama-3B 的 margin 在固定計分位置量（80% 輸出以 canonical 前綴開頭，
  其餘為代碼圍欄），絕對值解讀需谨慎；buy 決策 0/450 不受此影響。
- 未跑 order-swap 臂（recency 效應的跨模型驗證未做）。
