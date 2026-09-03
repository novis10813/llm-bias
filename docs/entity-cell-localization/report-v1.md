# Entity Cell Localization: V1 Discovery Report (E1 V1 + E2)

**Status:** V1 E1 discovery and E2 discovery complete. V1 E1 is a negative
result with an identified mechanism; E2 is descriptive. Calibration and
held-out test not run; V1 results are not re-labeled as any later version.

**Protocol:** [proposal-v1](proposal-v1.md). Version index: [README](README.md).
Later E1 version: [proposal-v2](proposal-v2.md) / [report-v2](report-v2.md).

## V1 runs on record

| run | state | note |
|---|---|---|
| `entity-cell-prepare-discovery-v1` | complete | 35 tickers, 420 header variants, 399 baseline, 105 financial prompts, 420 E2 donor contracts |
| `entity-cell-e1-smoke-v1` | failed (preserved) | metadata registration bug, fixed in `80b9a0c` |
| `entity-cell-e1-smoke-v2` | complete | one-ticker V1 smoke |
| `entity-cell-e1-discovery-v1` | complete | V1 discovery, all four E1 stages |
| `entity-cell-e2-discovery-v1`…`v4` | failed (preserved) | keyword-only attention hook, additivity tolerance, single-token continuation unpack, bf16-relative additivity scale — all fixed before v5 |
| `entity-cell-e2-discovery-v5` | complete | E2 attribution, selected-head readout, and patching contracts |

All runs under
`artifacts/qwen3.5-4b/entity-cell-localization/runs/`.

## E1 V1 discovery result (negative, mechanism identified)

- 31/35 Technology tickers share the same top-1 cell (L0, N4485); 4 share a
  second (L0, N5101). The V1 header family is template-dominated: all twelve
  variants share the identical three-line header, so header-reactive neurons
  have near-zero cross-variant standard deviation and dominate the stability
  score for every ticker.
- 0/35 tickers passed the amnesia eligibility gate.
- Known V1 limitation (found during V2 implementation): the localization-stage
  surface controls re-used the prepared `input_ids` of the original prompt
  instead of re-tokenizing the control prompt, so the reported
  `surface_control_summary` top-5 overlaps (5/5) are degenerate and not
  independent evidence. The template-domination conclusion rests on the
  cross-ticker top-1 collision and the amnesia gate. The V1 code path is left
  unchanged (frozen); E1 V2 implements re-tokenized controls from scratch.

## E2 discovery result (descriptive; E2 design is shared across E1 versions)

`entity-cell-e2-discovery-v5`, run under the V1 protocol (E2 is unchanged in
E1 V2). Selected heads (L31 H0/H1/H3, L19 H4, L27 H6) are all late
full-attention heads with instruction-dominant identity contribution
(mean_abs_id 0.002–0.014 versus instruction 0.02–0.11) and 1.00 consistency.
At the attention level, identity-header contribution to the Buy/Sell margin
is small relative to instruction context; consistent with the L16
instruction-context findings from the sector/context follow-up. Readout is
available for L19 H4 (105/105 buy>sell) and L27 H6; the three L31 heads are
readout-unavailable because the canonical lens covers source layers up to L30.
The e2-patching stage wrote patch contracts only; live head-output
intervention remains the explicit `patch_head_output` API (E3-B).

## Consequences

- E1 V1 produced no trusted candidate entity cell; the confirmation pipeline
  cannot be built from V1 selections (and per the V2 versioning boundary,
  confirmation may be built only from V2 discovery selections).
- The template-domination mechanism motivated the E1 V2 prompt-family change;
  see [report-v2](report-v2.md).
