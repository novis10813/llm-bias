# Concept-cone steering

Residual-stream steering experiments that test whether directions built from
company-level buy/sell preference differences (DIM, single neurons and
multi-dimensional concept cones) flip a decoder LLM's investment decision under
greedy generation.

Each finished task has a report in [`tasks/`](tasks/). [`AGENTS.md`](AGENTS.md)
describes the branch, task and file rules.

| Task | Report |
|---|---|
| T01 layer localization | [`tasks/T01-layer-localization/`](tasks/T01-layer-localization/) |

[`docs/concept-cone-steering/paper-vs-experiments.md`](docs/concept-cone-steering/paper-vs-experiments.md)
lists where the paper draft differs from the experiments.

## Setup

```bash
git clone https://github.com/anthropics/jacobian-lens.git third_party/jacobian-lens
uv sync
uv run pytest -q
```

Models are read from `.cache/models/<slug>`; run outputs go to
`artifacts/<model-slug>/`. Neither is tracked by git.
