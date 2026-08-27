# `llm_bias/static/` scope

這個目錄是 frozen entity-bias counterfactual dashboard 的無 build-step frontend；
archive 與還原規則見 [`../../AGENTS.md`](../../AGENTS.md)，counterfactual workflow/API
contract 見 [`../../../docs/archive/counterfactual-patching.md`](../../../docs/archive/counterfactual-patching.md)。

## 檔案責任

- `counterfactual.html`：頁面結構、pair/layer/mode/top-k controls，以及 source、
  target、patched 三個 grid panel。
- `counterfactual.js`：呼叫 `/api/info`、`/api/pairs`、`/api/counterfactual`，
  render grid、metrics、top-1 comparison，並在 hover 時顯示 cell 的完整 top-k。
- `counterfactual.css`：dashboard layout、三種狀態配色與 hover tooltip。

## Grid UI 語意

- column 是 input token position。
- row 是 Jacobian readout layer；`L15` 是 final output row，其餘是 fitted lens
  source layers。
- cell 內文顯示 top-1；hover tooltip 顯示 backend 回傳的所有 top-k token、機率、
  entropy 與 kurtosis。
- 不要把不同 token 的 top-1 probability 直接相減當成 causal effect；answer
  effect 應使用固定 source/target answer token 的 probability、logit margin 或
  normalized transfer。

## Instruction Index

本目錄沒有含 `AGENTS.md` 的直接子目錄。

## 前端修改後檢查

```bash
node --check archive/llm_bias/static/counterfactual.js
```

還原 frontend 後，再從 restored paths 執行 counterfactual workflow 的 regression tests。

若更新 static asset 的行為，請同步提高 `counterfactual.html` 中的 query-string
版本，避免長時間運行的瀏覽器使用舊版 JavaScript/CSS。
