# T02 protocol

T02 reports runs made before the task workflow existed, so this file does not hold a new design. It points to
the protocols as they stood when each run executed. Read them with the `git show` commands below.

| Run | Protocol | Version in effect | Committed before the run? |
|---|---|---|---|
| `confirmation-v1-freeze-20260925` (renderer, K, steer-suffix ids) | `docs/concept-cone-steering/confirmation-v1/proposal.md` | `git show e5b4a96:docs/concept-cone-steering/confirmation-v1/proposal.md` | Yes, same commit as the CPU run |
| `confirmation-v1-20260925-full-01`, arms `gates`, `ranking`, `alpha0`, `cal`, `dim`, `ops` | `docs/concept-cone-steering/confirmation-v1/proposal.md` and `docs/concept-cone-steering/operator-comparison-v2/proposal.md` | `git show 06efa5d:<path>` | Yes. Frozen before smoke, revised once in `ce5512d` (single-neuron target `mean_p d̂[p]`) before any full result |
| `confirmation-v1-supp-20261006-full-01`, arms `dim`, `ops` | None. The added doses are fixed in `SUPPLEMENT` of `scripts/probe_steering_confirmation_supplement.py` | `git show de2aae5:scripts/probe_steering_confirmation_supplement.py` | No. The doses were chosen on 2026-10-06 to fill the paper's dose grid, after the full-01 results were known |
