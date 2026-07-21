# Entity-level causal representation experiment

This repository studies when an entity-sensitive representation forms inside a
decoder language model and whether it causally changes the answer distribution.
The first experiment uses aligned factual counterfactual pairs and residual
activation patching. `jlens` is used as a Jacobian-lens readout; it is not
treated as a direct decoder of hidden chain-of-thought.

## Setup

Dependencies are managed with `uv`. The root project uses the local
`third_party/jacobian-lens/` and `third_party/jspace-viz/` as editable workspace
members. The external checkouts are intentionally ignored by the root Git
repository; their exact commits are recorded in `artifacts/visualization/` when
the integration server is started.

On a fresh checkout, restore the ignored external worktrees before syncing:

```bash
mkdir -p third_party
git clone https://github.com/anthropics/jacobian-lens.git third_party/jacobian-lens
git clone https://github.com/Festyve/jspace-viz.git third_party/jspace-viz
uv sync
```

The implementation targets `unsloth/Llama-3.2-1B-Instruct`. Start the model
download in the background with:

```bash
mkdir -p artifacts .cache/models
nohup uv run hf download unsloth/Llama-3.2-1B-Instruct \
  --local-dir .cache/models/llama-3.2-1b-instruct \
  > artifacts/llama_download.log 2>&1 &
```

## Run

```bash
uv sync
uv run pytest
uv run python -m llm_bias prepare-data
uv run python -m llm_bias fit-lens --calibration-prompts 16 --layer-stride 4
uv run python -m llm_bias run-patch --lens artifacts/entity_control/jacobian_lens.pt
uv run python -m llm_bias visualize
uv run python -m llm_bias serve-viz --lens artifacts/entity_control/jacobian_lens.pt
```

The last command starts a local interactive counterfactual dashboard at
`http://127.0.0.1:8321`. It runs source, target, and patched forwards for the
selected pair and displays their J-space top-1 readouts. The server writes the
two ignored external checkout revisions to
`artifacts/visualization/dependencies.json`.

The smoke run can use `--max-pairs 4` and two calibration prompts. The full
current run produced 170 aligned pairs across countries, months, animals, and
numbers. Results are written under `artifacts/entity_control/`.

The visualization command writes the corrected transfer curve,
`patch_transfer_heatmap.png`, and `jacobian_readout_heatmap.png`.
