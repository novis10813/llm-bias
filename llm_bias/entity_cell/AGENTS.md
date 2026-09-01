# `llm_bias/entity_cell/` scope

This package owns the frozen entity-cell preparation contract and the implemented
V1 lifecycle, discovery summary, and confirmation evaluation. It may use shared
`core.prompt_input`, `core.analysis`, and `core.artifacts` mechanics, but it must not
import another experiment package. Preparation loads a tokenizer only; model stages
emit compact JSONL/JSON and never persist tensors or runtime activations. E2 donor
contracts stay experiment-owned and bind the four frozen identity conditions.

Local verification:

```bash
uv run pytest -q tests/test_entity_cell_preparation.py tests/test_entity_cell_e1.py tests/test_entity_cell_e2.py tests/test_entity_cell_e3.py tests/test_entity_cell_readout.py
uv run python -m compileall -q llm_bias/entity_cell
```
