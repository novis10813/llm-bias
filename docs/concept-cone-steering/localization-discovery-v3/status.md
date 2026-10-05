# Discovery-v3 discovery runs are executing on GLM, Qwen and Gemma

**State:** `scripts/run_stance_localization_discovery_v3.py` and `llm_bias/core/inference/stance_localization_batched.py` implement the [proposal](proposal.md) for GLM/Qwen/Gemma. GPT fails before loading inputs. No localization effect is claimed yet. All outputs remain `research_eligible=false`.

| Model | Discovery cells | Validation cells |
|---|---:|---:|
| qwen3.5-4b | 6,144 | ≤ 4,800 |
| glm4-9b-0414 | 6,144 | ≤ 4,800 |
| gemma4-12b-it | 7,168 | ≤ 4,800 |

**Verified on CPU:** new batched-executor and runner suites 25 passed. All localization, structured-output, Harmony, no-op, intervention and transient-capture suites 1293 passed / 67 skipped. `uv lock --check` and `git diff --check` pass. The full `uv run pytest -q` was not run.

**R1:** Fake-model equality is float64. It proves per-row hooks, grammar state, stopping and records match batch-one logic. It does not show that BF16 GPU rows match batch-one. The per-call control row measures that on real runs.

**Running (commit `2e4f321`, idlab):** job 52 GLM on GPU0, job 55 Qwen on GPU2, job 56 Gemma on GPU3. Output roots: `artifacts/<slug>/concept-cone-steering/runs/localization-discovery-v3-stance-rb260930-<job>/`. GLM runs about 30–45 s per pair. Job 49 (commit `d848335`, before batched masking) was stopped after 39/256 pairs. Its partial records stay in the idlab worktree and are not used.

**F1:** On the first GLM batches every control row differs from the batch-one clean target in reason text, starting around token 34–44, while all control decisions match. Full-output `control_match` is therefore near 0. Decision-level control agreement can be recomputed from `records/batch_*.json`.

**F2:** A GPU profile (jobs 50/51, GPU1) showed per-row xgrammar mask filling, about 41 ms per row per step, dominating batched decode. Batched masking cut 25-row decode from 1,759 to 286 ms/step and one GLM pair from 129 s to 45 s, including a one-time 20 s triton compile.

**A1:** After discovery completes, run validation per model with `--discovery-run` pointing at that model's discovery root.
