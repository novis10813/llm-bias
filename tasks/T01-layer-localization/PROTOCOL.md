# T01 protocol

T01 reports runs made before the task workflow existed, so this file does not hold a new design. It points to
the protocols as they stood when each run executed. Read them with the `git show` commands below. They are not
copied here, because committed protocols are never rewritten.

| Run | Protocol | Version in effect | Committed before the run? |
|---|---|---|---|
| `phase2b-v2-427-01` (Stage 1) | `docs/balanced-evidence-gap/details/proposal-phase2-v2.md` | `git show 7bf2e8f:docs/balanced-evidence-gap/details/proposal-phase2-v2.md` | No. The four runs finished 2026-09-23 to 2026-09-24 00:19 UTC, and the protocol was first committed with the code in `7bf2e8f` (2026-09-24 00:31 UTC). It is marked development, and its layer-selection rule was not fixed in advance |
| `audit-v1-20260925` (overlap exclusion, A2) | `docs/concept-cone-steering/c2-v3-steering-prompt/proposal.md`, section "C2 可寫範圍與 A2" | `git show e5b4a96:docs/concept-cone-steering/c2-v3-steering-prompt/proposal.md` | Yes, same commit as the CPU run |
| `confirmation-v1-20260925-full-01`, arms `c2v3` and `c2v3_gen` | `docs/concept-cone-steering/c2-v3-steering-prompt/proposal.md` under `docs/concept-cone-steering/confirmation-v1/proposal.md` | `git show 06efa5d:docs/concept-cone-steering/c2-v3-steering-prompt/proposal.md` | Yes. Frozen before smoke, last revised in `ce5512d` before any full result |
