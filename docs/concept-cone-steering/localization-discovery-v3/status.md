# Discovery-v3 runner is implemented and CPU-tested, no GPU run yet

**State:** `scripts/run_stance_localization_discovery_v3.py` and `llm_bias/core/inference/stance_localization_batched.py` implement the [proposal](proposal.md) for GLM/Qwen/Gemma. GPT fails before loading inputs. No job has been submitted and no localization effect is claimed. All outputs remain `research_eligible=false`.

| Model | Discovery cells | Validation cells |
|---|---:|---:|
| qwen3.5-4b | 6,144 | ≤ 4,800 |
| glm4-9b-0414 | 6,144 | ≤ 4,800 |
| gemma4-12b-it | 7,168 | ≤ 4,800 |

**Verified on CPU:** new batched-executor and runner suites 25 passed. All localization, structured-output, Harmony, no-op, intervention and transient-capture suites 1293 passed / 67 skipped. `uv lock --check` and `git diff --check` pass. The full `uv run pytest -q` was not run.

**R1:** Fake-model equality is float64. It proves per-row hooks, grammar state, stopping and records match batch-one logic. It does not show that BF16 GPU rows match batch-one. The per-call control row measures that on real runs.

**A1:** First GPU run: GLM discovery with fresh run root `artifacts/glm4-9b-0414/concept-cone-steering/runs/localization-discovery-v3-<job-id>/`. Read the `control_match` rate from `records/batch_*.json` after the first pairs, before launching Qwen/Gemma.
