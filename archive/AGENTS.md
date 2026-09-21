# `archive/` guidance

本目錄保存 frozen counterfactual、synthetic、10-K/8-K 與 financial-soundness code、
tests、scripts。不要在這裡開發新功能。若 active baseline experiment 需要舊實作，先評估能否在 active tree
重寫成較小、符合 shared workflow contract 的版本；確需沿用時，依
[`README.md`](README.md) 還原完整 package、tests、scripts、docs 與 entry point。

## Archive rules

- 保留 archived public API、artifact schema 與 research semantics，讓既有 artifacts
  可解讀。
- Active shared core 變更若破壞 restoration，更新 `README.md` 的 dependency 或 restore
  steps；不要只在 frozen package 加 compatibility patch。
- Archived tests 不在 root pytest `testpaths`。只有還原 workflow 或查核 archived
  contract 時才執行，並在結果中標明 restore state。
- 對應操作文件與索引位於 [`../docs/archive/README.md`](../docs/archive/README.md)。

## Instruction Index

- [`llm_bias/AGENTS.md`](llm_bias/AGENTS.md)：frozen packages 與其局部 frontend 入口。

`tests/` 或 `scripts/` 若在還原前形成各自獨立規範，再新增局部 `AGENTS.md` 並只更新
本節。

## Verification

Archive-only 文件修改至少檢查相對連結。實際還原 package 時，依 `README.md` 搬回
active paths，再執行 root 完整驗證；不要把 archive 內的原始 path 當成 installed
entry point。
