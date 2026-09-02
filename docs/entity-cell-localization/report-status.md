# Entity Cell Localization and Downstream Attribution: Status

**Status:** V1 discovery and E2 discovery complete (negative / descriptive
findings); E1 V2 discovery complete (1/35 trusted candidate entity cell);
confirmation not frozen.

**Protocol:** V1 frozen in [the proposal](proposal.md); E1 V2 frozen in the
proposal's "E1 V2: surface-varying localization" section.

## Runs on record

| run | state | note |
|---|---|---|
| `entity-cell-prepare-discovery-v1` | complete | 35 tickers, 420 header variants, 399 baseline, 105 financial prompts, 420 E2 donor contracts |
| `entity-cell-prepare-discovery-v2` | complete | same inputs plus 420 frame variants and the template-only control (`--localization-family v2-frames`) |
| `entity-cell-e1-smoke-v1` | failed (preserved) | metadata registration bug, fixed in `80b9a0c` |
| `entity-cell-e1-smoke-v2` | complete | one-ticker V1 smoke |
| `entity-cell-e1-smoke-v3` | failed (preserved) | CUDA OOM at startup (GPU full) |
| `entity-cell-e1-smoke-v4` | complete | one-ticker V2 smoke (CPU) |
| `entity-cell-e1-discovery-v1` | complete | V1 discovery, all four E1 stages |
| `entity-cell-e1-discovery-v2` | complete | V2 discovery, all four E1 stages (GPU 0, bf16, ~37 min) |
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
frame variants, template-only control, all V1 content reused).

### E1 V2 discovery result (`entity-cell-e1-discovery-v2`)

- **1/35 tickers passed all four frozen gates: FTNT.** Its top-1 cell (L0,
  N104, stability score 363.4) is form-robust (absent from both frame-family
  surface-control top-5s), not in the template signature, retained in the
  held set (top-5 overlap 1, held rank 4 via its rank-2 cell L1 N575), and
  passed the amnesia endpoint gate on 2/3 eligible financial prompts at
  $\\alpha=-3$ (target progress +0.067 / +0.265, each above both the
  wrong-entity and matched-random controls; the third prompt fails with
  negative target progress). No ticker had a degenerate wrong-entity control.
- **Caveat (cross-ticker selectivity):** FTNT's top-1 cell (L0, N104) is the
  shared top-1 of 17/35 tickers in the frame family; the other 16 share it
  but fail form-robust (their surface controls fire on it). The gates are
  per-ticker, so the pass is a boundary case: (L0, N104) is a candidate,
  not yet a demonstrated FTNT-specific identity cell. Selective
  intervention (E3) is the next test of entity specificity.
- **Structure:** V2 top-1 collision is (L0, N104)×17, (L0, N5101)×6,
  (L0, N4485)×4, then singletons — template neuron (L0, N4485) dropped from
  31/35 (V1) to 4/35; 25/35 tickers have zero frame∩header top-5 overlap,
  5 tickers overlap 3, 5 overlap 4. The template-only signature top-5 is
  led by (L0, N4485) (|z| 277), reproducing the V1 collision neuron as the
  template-reactive landmark; 10/35 tickers have a candidate in the
  signature. Only 1/35 tickers is form-robust; 9/35 pass the amnesia
  endpoint gate alone (AKAM, AMAT, CTSH, FFIV, FTNT, IBM, JKHY, MU, TXN),
  all but FTNT excluded by form-robust, template, or held overlap.

## Next steps

1. Decide the V2 confirmation design: the single trusted candidate (FTNT,
   L0 N104) must be handled explicitly — per-ticker confirmation with a
   pre-registered cross-ticker specificity analysis, or a V3 design with a
   selectivity gate. Any change to gates/selection requires a new frozen
   version per experiment-versioning.
2. If confirmed: calibration (12 tickers) → test (11 tickers) only after a
   passing calibration gate.
3. E3 downstream (selected-head intervention) can proceed from the E2 v5
   selections (L31 H0/H1/H3, L19 H4, L27 H6); E3 upstream can target the
   FTNT candidate if the confirmation design authorizes it.
