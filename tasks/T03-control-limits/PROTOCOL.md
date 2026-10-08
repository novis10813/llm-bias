# T03 protocol

T03 reports runs made before the task workflow existed, so this file does not hold a new design. It points to
the protocols as they stood when each run executed. Read them with the `git show` commands below.

| Run | Protocol | Version in effect | Committed before the run? |
|---|---|---|---|
| `confirmation-v1-20260925-full-01`, arms `alpha0`, `cal`, `dim`, `random`, `jitter` | `docs/concept-cone-steering/confirmation-v1/proposal.md` and `docs/concept-cone-steering/operator-comparison-v2/proposal.md` (random, jitter, C5 rule) | `git show 06efa5d:<path>` | Yes. Frozen before smoke, revised once in `ce5512d` before any full result |
| `confirmation-v1-20260925-full-01`, arms `evidence`, `anon` | `docs/concept-cone-steering/evidence-sensitivity-v1/proposal.md` (conditions, flip-dose contrasts, anonymous identities) | `git show 06efa5d:docs/concept-cone-steering/evidence-sensitivity-v1/proposal.md` | Yes |
| `confirmation-v1-supp-20261006-full-01`, arms `dim`, `random`, `evidence`, `anon` | None. The added doses are fixed in `SUPPLEMENT` of `scripts/probe_steering_confirmation_supplement.py` | `git show de2aae5:scripts/probe_steering_confirmation_supplement.py` | No. The doses were chosen on 2026-10-06 to fill the paper's validation table, after the full-01 results were known |
