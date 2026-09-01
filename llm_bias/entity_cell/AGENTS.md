# `llm_bias/entity_cell/` scope

This package owns the frozen entity-cell preparation contract. It may use shared
`core.prompt_input` and `core.artifacts` mechanics, but it must not import another
experiment package or implement E1/E2/E3 model execution. Preparation loads a
tokenizer only, emits compact JSONL and metadata, and never persists tensors or
runtime activations.

Local verification:

```bash
uv run pytest -q tests/test_entity_cell_preparation.py
uv run python -m compileall -q llm_bias/entity_cell
```
