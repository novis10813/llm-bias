# `archive/llm_bias/` guidance

本目錄保存已從 active `llm_bias/` 移出的 frozen packages。Package 清單、相依關係、
還原順序與研究語意見 [`../README.md`](../README.md)；本檔只補充 package 層入口。

## Package boundaries

- `counterfactual_data/` 依賴 `counterfactual_patching.data`。
- `ten_k_change_data/` 依賴 active `llm_bias.prompt_analysis`。
- 其餘 frozen packages 只依賴 active `llm_bias.core`。
- 不從 active packages import 這些 archived modules。還原時先恢復 package 與 entry
  point，再更新 active workflow boundary tests 和文件。

## Instruction Index

- [`static/AGENTS.md`](static/AGENTS.md)：counterfactual dashboard 的 frozen frontend
  與 grid 語意。

若某個 frozen package 在還原後形成新的局部規範，應在 restored active path 建立
`AGENTS.md`；不要把新規則留在 archive copy。
