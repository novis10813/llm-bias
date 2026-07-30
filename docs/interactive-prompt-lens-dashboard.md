# Interactive Prompt Lens Dashboard

Prompt Lens Dashboard 讓使用者輸入任意 prompt，查看 Qwen 每個 input token、
每個 decoder layer 的 transported Jacobian-lens readout，並可同時生成模型的
實際 greedy response。

這個 dashboard 與 counterfactual-patching dashboard 是不同入口：

- Prompt Lens：單一任意 prompt 的逐層 readout 與實際 response。
- Counterfactual dashboard：固定 source/target pair 的 source、target、patched
  三種狀態與 causal transfer metrics。

## 啟動方式

Qwen 預設使用 model-specific canonical lens：

```bash
uv run prompt-analysis serve \
  --model .cache/models/qwen3.5-4b \
  --host 0.0.0.0 \
  --port 8322
```

瀏覽器開啟 `http://127.0.0.1:8322`。若要測試非 canonical experimental lens，
可明確指定：

```bash
uv run prompt-analysis serve \
  --model .cache/models/qwen3.5-4b \
  --lens artifacts/candidate_lenses/qwen3.5-4b/mixed/jacobian_lens.pt
```

Server 啟動時只載入一次 model 與 lens。它會檢查 hidden width、model metadata
與完整逐層 coverage；缺 layer 或放錯 model 的 lens 會在提供服務前被拒絕。

## 操作與輸出

主要 controls：

| Control | 語意 |
|---|---|
| Prompt | 任意非空文字，最多 20,000 characters |
| Mode: `jlens` | 使用 fitted Jacobian transport 解讀 intermediate residual |
| Mode: `logit` | 對各層 residual 直接套用 unembedding，作為比較 baseline |
| Top-k | 每個 layer/token cell 回傳 1–32 個 token |
| Max sequence length | 8–2048；超長 input 會截斷並標記 |
| Chat template | 把輸入格式化成一個 user turn 與 generation prompt |
| Enable thinking | 只在 tokenizer template 支援時啟用 |
| Generate response | 對完全相同的 formatted prompt 執行 deterministic greedy generation |
| Max new tokens | 1–256 |

Grid 的 column 是 input token position，row 是 model layer。每個 cell 顯示 top-1
readout，hover 顯示完整 top-k。Dashboard 也回傳 input prompt、實際 formatted
prompt、token IDs/text、prompt length、是否截斷、fitted layer coverage 與 lens
calibration prompt count。

勾選 Generate response 後，頁面額外顯示模型從同一份 formatted prompt 生成的
實際文字與 token count。例如輸入 `Do you love me?` 時，可以同時比較逐層
J-lens readout 與模型真正回答的 continuation。

## HTTP API

### `GET /api/info`

回傳常駐 model/lens 狀態，包括：

- model id、device、layer count 與 hidden width；
- fitted layers、缺失 layers 與 expected layer count；
- lens calibration prompt count 與 artifact path；
- chat template/thinking support；
- server control limits。

### `POST /api/readout`

Request：

```json
{
  "prompt": "Do you love me?",
  "mode": "jlens",
  "top_k": 8,
  "max_seq_len": 256,
  "chat": true,
  "enable_thinking": false,
  "generate_continuation": true,
  "max_new_tokens": 64
}
```

Response 由 `jspace-viz` compact grid 加上以下欄位組成：

- `input_prompt`：使用者原始輸入；
- `prompt`：實際送入 model 的 formatted prompt；
- `continuation`：optional 實際 greedy response；
- `response_token_count`；
- `chat_enabled`、`thinking_enabled`；
- `requested_max_seq_len`、`truncated`。

Prompt validation errors 回傳 HTTP 400；model/readout runtime errors 回傳 HTTP
500 並寫入 server log。

## 計算與保存邊界

每個 request 即時計算 readout。Backend 只回傳 compact token IDs、decoded text、
top-k probabilities 與必要 metadata，不把完整 raw activations 寫入磁碟。
模型 response 使用 `do_sample=False`，因此相同 model、prompt 與 generation
settings 下可重現。

## 研究解讀限制

- Jacobian-lens cell 是 transported readout，不是模型公開的 chain-of-thought。
- 不同 cell 的 top-1 token 不能直接串成離散 reasoning trace。
- `logit` mode 是 direct-unembedding baseline，不等同 fitted Jacobian transport。
- Dashboard 的逐層可讀性屬 representation evidence，不單獨證明 causal effect。
- Entity bias 結論仍需要固定 outcome margin、activation patch、matched controls
  與 paired statistics。

因此 Prompt Lens 最適合用於 qualitative inspection、debugging 與挑選後續
confirmatory cases；正式 bias 結論應來自 batch counterfactual experiment。
