# Entity Cell Localization and Downstream Attribution: Status

**Status:** V1 discovery and E2 discovery complete (negative / descriptive
findings); E1 V2 (surface-varying localization) implemented and prepared;
V2 discovery not yet run; confirmation not frozen.

**Protocol:** V1 frozen in [the proposal](proposal.md); E1 V2 frozen in the
proposal's "E1 V2: surface-varying localization" section.

## Runs on record

| run | state | note |
|---|---|---|
| `entity-cell-prepare-discovery-v1` | complete | 35 tickers, 420 header variants, 399 baseline, 105 financial prompts, 420 E2 donor contracts |
| `entity-cell-prepare-discovery-v2` | complete | same inputs plus 420 frame variants and the template-only control (`--localization-family v2-frames`) |
| `entity-cell-e1-smoke-v1` | failed (preserved) | metadata registration bug, fixed in `80b9a0c` |
| `entity-cell-e1-smoke-v2` | complete | one-ticker V1 smoke |
| `entity-cell-e1-discovery-v1` | complete | V1 discovery, all four E1 stages |
| `entity-cell-e2-discovery-v1`…`v4` | failed (preserved) | keyword-only attention hook, additivity tolerance, single-token continuation unpack, bf16-relative additivity scale — all fixed before v5 |
| `entity-cell-e2-discovery-v5` | complete | E2 attribution, selected-head readout, and patching |

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
  cross-ticker top-1 collision and the amnesia gate. The V1 code is left
  unfrozen-unchanged; V2 implements re-tokenized controls from scratch.

## E2 discovery result (descriptive)

Selected heads (L31 H0/H1/H3, L19 H4, L27 H6) are all late full-attention
heads with instruction-dominant identity contribution
(mean_abs_id 0.002–0.014 versus instruction 0.02–0.11) and 1.00 consistency.
At the attention level, identity-header contribution to the Buy/Sell margin is
small relative to instruction context; consistent with the L16
instruction-context findings from the sector/context follow-up.

## E1 V2 (surface-varying localization)

Implementation (branch `feat/entity-cell-e1v2`): frozen twelve natural-sentence
frames (F0–F7 localization, H0–H3 held), frame-family surface controls
(`anonymous_name_frames`, `name_form_control_frames`) rendered and
re-tokenized at run time, the template-only control prompt with a frozen
template signature (top-5 by absolute z-score), the V1 header family re-run
for comparison only, the deterministic non-degenerate wrong-entity rule with a
`degraded_control` flag, and the four V2 gates (held overlap, form-robust,
template-robust, amnesia endpoint). V1 prepared outputs remain
byte-identical under the default `v1-header` family.

Preparation is complete: `entity-cell-prepare-discovery-v2` (35 tickers, 420
frame variants, template-only control, all V1 content reused). V2 discovery
requires a GPU with ~10 GB headroom; it is not yet run.

## Next steps

1. Run E1 V2 discovery (all E1 stages) on a free GPU; analyze trusted-candidate
   count, template/header-family overlap, and degenerate-control flags.
2. If V2 yields trusted cells: freeze confirmation from V2 selections,
   calibration (12 tickers) → test (11 tickers) only after a passing
   calibration gate.
3. E3 downstream (selected-head intervention) can proceed from the E2 v5
   selections; E3 upstream remains blocked until trusted cells exist.
