# Archived workflow documents

本目錄保存 frozen counterfactual、synthetic 與 10-K/8-K workflow 的操作與 artifact
語意。Active `pyproject.toml` 不註冊這些 CLI。若要執行命令，先依
[`archive/README.md`](../../archive/README.md) 還原對應 package、tests、scripts 與
entry point。

| Document | Frozen implementation | Scope |
|---|---|---|
| [Counterfactual patching](counterfactual-patching.md) | `archive/llm_bias/counterfactual_patching/` | residual span patching、controls、dashboard |
| [8-K counterfactual dataset](counterfactual-dataset-generation.md) | `archive/llm_bias/counterfactual_data/` | annotation、review、promotion、entity-only pairs |
| [EDGAR 8-K preparation](edgar-8k-preparation.md) | `archive/llm_bias/edgar_preparation/` | filing cleaning 與 event staging corpus |
| [Synthetic entity-bias pilot](synthetic-entity-bias.md) | `archive/llm_bias/synthetic_entity_bias/` | synthetic task contract、localization、artifacts |
| [10-K metadata-change dataset](ten-k-change-dataset.md) | `archive/llm_bias/ten_k_change_data/` | adjacent filing metadata-change windows |
| [Easy-bias binary association](easy-bias-binary-association.md) | restored `counterfactual_patching` CLI | exploratory Traditional Chinese association workflow |

這些文件保留 existing artifacts 的 normative contract。修改 active shared core 後，若
archive restoration 受到影響，應更新 `archive/README.md` 的 dependency/restoration
說明；不要在 frozen package 內開發新功能。
