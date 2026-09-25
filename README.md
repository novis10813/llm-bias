# Concept-cone steering

Residual-stream steering experiments that test whether directions built from
company-level buy/sell preference differences (DIM and multi-dimensional concept
cones) flip a decoder LLM's investment decision under greedy generation.

Research notes, versioned protocols, status, and figures live in
[`docs/concept-cone-steering/`](docs/concept-cone-steering/). Start with
[`claim-to-evidence.md`](docs/concept-cone-steering/claim-to-evidence.md).

Earlier research lines were removed from the working tree; they are preserved
at git tag `pre-cleanup`.

## Setup

```bash
git clone https://github.com/anthropics/jacobian-lens.git third_party/jacobian-lens
uv sync
uv run pytest -q
```

Models are read from `.cache/models/<slug>`; run outputs go to
`artifacts/<model-slug>/`. Neither is tracked by git.
