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
| T02 steering operators | [`tasks/T02-steering-operators/`](tasks/T02-steering-operators/) |
| T03 control limits | [`tasks/T03-control-limits/`](tasks/T03-control-limits/) |
| T04 stance rebuild | [`tasks/T04-stance-rebuild/`](tasks/T04-stance-rebuild/) (code only, no report) |
| T05 single-neuron dial | Stopped, no report |

The T01–T03 READMEs list where the paper draft differs from their experiments (P01–P20).

## Setup

```bash
git clone https://github.com/anthropics/jacobian-lens.git third_party/jacobian-lens
uv sync
uv run pytest -q
```

Models are read from `.cache/models/<slug>`; run outputs go to
`artifacts/<model-slug>/`. Neither is tracked by git.
