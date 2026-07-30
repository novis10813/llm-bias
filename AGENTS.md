# 環境需求

- 使用 `uv` 管理 python 虛擬環境
- `uv add` 添加套件

# Repository scope

這份文件適用於整個 repository；子目錄中的 `AGENTS.md` 會補充更具體的
模組規則。

## 專案目的

本 repo 實驗 entity-level bias 如何在 decoder LLM 的 residual stream 中
形成，以及把 source entity 的 activation 替換成 target entity 後，是否會
改變最終答案分布。

主要流程是：

1. 從 experiment spec 建立 source/target entity-span counterfactual pairs。
2. 使用 `jlens` fitting Jacobian lens。
3. 執行 residual activation patching，計算 answer-margin transfer。
4. 使用 `jspace-viz` 的 model/lens 格式與 grid 概念呈現 source、target、patched
   三種狀態。

## Next steps / 研究 roadmap

目前第一優先是完成 span-level activation patch；後續實驗必須沿用以下順序。

### 1. Span patch（目前工作）

- `Pair` 同時保留 entity 的 token start/end 與完整 token-id span；舊版
  single-token `pairs.jsonl` 必須可載入。
- source/target entity span 可以是不同長度，batch patch 使用 normalized
  span-internal token centers，對每個 source position 選 nearest target
  position；不可直接插入或刪除 sequence token，也不合成 activation。
- source/target prompt 可以有不同 token 長度；answer logits 各自讀取各自的
  final position。control patch 使用兩邊各自最後一個非-entity position。
- batch analysis 必須輸出 source span、target span、position mapping 與 mapping
  strategy；不可保存完整 raw activations。
- interactive dashboard 暫時只接受等長 source/target prompt；不同長度 pair
  必須回傳明確 validation error，待後續 token-position alignment schema 再支援。

### 2. Bias-specific counterfactual data

- 新增相同 headline/context、只替換 entity identity 的 pairs：真實名稱、匿名
  entity、matched synthetic entity。
- 每個 content 建立以下四種 contrast，並在結果中分開報告：
  - `real vs real`：測量 entity-specific prior 差異。
  - `real vs anonymous`：測量移除已知 entity identity 後的效果。
  - `real vs synthetic`：區分 memorized identity 與名稱形式差異。
  - `synthetic vs synthetic`：作為 token/name-form baseline。
- entity 效果不能與 factual knowledge、headline wording 或 tokenization 差異
  混淆；需記錄 condition、content id、entity id 與預期 outcome。
- source/target pair 必須共用完全相同的 headline/context 與 expected outcome；
  bias experiment 不應因 entity 改變正確答案。
- 第一版優先建立等長、single-token 或等長 multi-token names，確保可直接在
  dashboard 比較；variable-length pairs 仍由 batch span patch 支援。
- bias pairs 的 outcome 通常相同，不能套用 factual `source_answer →
  target_answer` transfer；需使用固定 outcome options 的 logit margin，例如
  `logit(high) - logit(low)`，並記錄 source、target、patched 三者分數。
- 修正或排除 upstream factual spec 中不可靠的答案 mapping，再將結果用於
  bias 結論。

### 3. Representation readout

- 每個 layer 記錄 source/target residual distance、entity-position 的
  Jacobian-lens readout divergence，以及 answer-position 的 outcome logit/
  margin difference。
- Jacobian lens 是 transported readout，不可直接描述為模型的 chain-of-
  thought 或離散 reasoning path。

### 4. Causal controls and statistics

- 執行 source→target 與 target→source 雙向 patch，並比較 unrelated/random
  entity、非-entity position 與 residual interpolation controls。
- 使用 paired bootstrap 與 permutation test；報告 direct entity effect、
  representation signal、causal transfer 三種結果，不把其中一種單獨當成 bias
  證明。

### Span patch acceptance criteria

- 覆蓋 1→1、1→2、2→1、2→3 等 span mapping 的 deterministic tests。
- tensor、tuple、list、model-output decoder block output 都保留原本的 tail。
- variable-length source/target pair 能完成 batch logits patch；舊版單 token
  pair 與等長 visualization 行為不得退化。
- 完成後執行 `uv run pytest -q`、`uv run python -m compileall -q llm_bias`、
  `uv lock --check` 與 `uv build`。

## 依賴與外部 checkout

- Python 依賴與 lockfile 一律由 `uv` 管理；新增套件使用 `uv add`。
- `third_party/jacobian-lens` 與 `third_party/jspace-viz` 是 editable workspace
  members，但整個 `third_party/` 被 `.gitignore` 忽略。
- 新環境請依照 `README.md` clone 兩個外部 repo 後再執行 `uv sync`。
- 不要把模型權重、patch 結果、lens binary 或第三方 checkout 加入 root Git
  repository。這些內容位於 `.cache/`、`artifacts/`、`third_party/`。
- 每個 model 只有一個 active、完整逐層的 canonical lens：
  `artifacts/lenses/<model>/jacobian_lens.pt`。Partial/stride 實驗與 fitting
  checkpoint 必須放在 `artifacts/archive/`，不可混入 active model folder。

## 常用驗證

```bash
uv lock --check
uv run pytest -q
uv run python -m compileall -q llm_bias
uv build
```

視覺化 server：

```bash
uv run counterfactual-patching serve \
  --host 0.0.0.0 \
  --port 8321 \
  --pairs artifacts/counterfactual_patching/pairs.jsonl
```

# 外部套件與工具

## jacobian-lens
Anthropic 於 2026 年推出的一項大語言模型（LLM）可解釋性工具。它能「解碼」模型在各個隱藏層（中間層）的內部神經元活動，直接翻譯出該狀態即將輸出的概念，進而揭示 AI 模型內部的「全局工作空間 (Global Workspace)」與隱性推理過程。
