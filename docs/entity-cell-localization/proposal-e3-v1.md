# Entity Cell Localization: Proposal E3 V1 (Upstream Suppression with Cross-Ticker Specificity and Downstream Component Attenuation)

**Document status:** frozen E3 V1 Discovery protocol; implemented; ready for first discovery inference run. Version index: [README](README.md).

**Model for run:** Qwen3.5-4B (`.cache/models/qwen3.5-4b`)  
**Depends on:**
- Prepared inputs: `artifacts/qwen3.5-4b/entity-cell-localization/runs/entity-cell-prepare-discovery-v2` (contains 105 financial prompts across 35 Technology discovery tickers, 3 prompts/ticker).
- Completed E1 V2 discovery: `artifacts/qwen3.5-4b/entity-cell-localization/runs/entity-cell-e1-discovery-v2` (sole trusted candidate: `FTNT`, cell `(L0, N104)`, stability score 363.4; baseline stats in `e1/baseline_stats.json`).
- Completed E2 discovery: `artifacts/qwen3.5-4b/entity-cell-localization/runs/entity-cell-e2-discovery-v5` (selected heads: `(31, 0)`, `(31, 1)`, `(31, 3)`, `(19, 4)`, `(27, 6)`).

---

## Motivation & Scientific Tension

E1 V2 natural-sentence frame localization identified **FTNT (Fortinet)** as the sole ticker passing all four frozen gates (held-frame retention, form-robust, template-robust, and amnesia endpoint). Its candidate entity cell is **`(L0, N104)`**.

However, V2 discovery also revealed a major caveat: **17 out of 35 Technology tickers share `(L0, N104)` as their top-1 localization neuron**. The other 16 tickers failed form-robustness because their surface-form controls (`anonymous_name_frames` or `name_form_control_frames`) also activated `(L0, N104)`.

This raises a fundamental mechanistic question:
> **Is `(L0, N104)` a genuine company identity cell specific to FTNT, or is it a generic "company entity-slot / proper-noun detector" that activates across many corporate names and passed on FTNT merely by boundary thresholding?**

Observational localization alone cannot resolve this. **Causal suppression** can:
1. If suppressing `(L0, N104)` moves FTNT's decision margin toward its Anonymous baseline while leaving same-collision peers (`ADI`, `MU`) unaffected, then `(L0, N104)` exhibits genuine **causal identity specificity**.
2. If suppressing `(L0, N104)` moves `ADI` and `MU` toward Anonymous with comparable magnitude, then `(L0, N104)` is a **generic entity-slot neuron**. This would provide definitive evidence that Qwen3.5-4B does not localize individual company identities to single monosemantic early MLP neurons.

---

## Experiment Design

Phase E3 separates two causal interventions:
- **E3-A**: upstream entity-cell suppression with within-ticker controls and cross-ticker specificity controls.
- **E3-B**: downstream component attenuation on the E2-selected attention heads.

### Population & Prompts

| Role | Ticker | Localization top-1 | Selection rationale | Prompts |
|---|---|---|---|---|
| **Target ticker** | `FTNT` | `(0, 104)` | Sole trusted candidate from E1 V2 | 3 financial prompts |
| **Peer control 1 (same collision)** | `ADI` | `(0, 104)` | Shared top-1, failed form-robust | 3 financial prompts |
| **Peer control 2 (same collision)** | `MU` | `(0, 104)` | Shared top-1, failed form-robust; passed amnesia alone | 3 financial prompts |
| **Peer control 3 (different top-1)** | `FTV` | `(0, 5101)` | Non-104 top-1 in L0 | 3 financial prompts |

Total prompts evaluated: 4 tickers × 3 financial prompts = 12 prompts.

### Phase E3-A: Upstream Entity-Cell Suppression

For each prompt, scale the pre-`down_proj` activation of the designated neuron across the specified scope using the fixed dose grid:
$$\alpha \in \{1.0, 0.5, 0.0, -1.0, -2.0, -3.0\}$$

#### Conditions evaluated in E3-A:

1. **Target Ticker Arm (`FTNT`)**:
   - `target`: suppress `(0, 104)`.
   - `wrong_entity`: suppress `(0, 5101)` (derived deterministically as the top candidate of the alphabetically next ticker `FTV` in the discovery split from `cells.jsonl`).
   - `matched_random`: suppress a same-layer (L0) neuron sampled via `select_matched_random_neuron` from baseline stats.

2. **Cross-Ticker Specificity Arms (`ADI`, `MU`, `FTV`)**:
   - `target_cell_cross_ticker`: suppress the **exact same target cell `(0, 104)`** on the peer ticker's prompt.
   - `matched_random`: suppress a same-layer (L0) neuron matched to the peer's baseline distribution.

#### Scopes:
- `all_positions` (primary): scales pre-`down_proj` across the full input sequence (Barzilay et al. protocol).
- `header_only` (secondary): scales pre-`down_proj` only over the `identity_header` token span.

### Phase E3-B: Downstream Component Attenuation

For the 5 E2-selected full-attention heads (`(31, 0)`, `(31, 1)`, `(31, 3)`, `(19, 4)`, `(27, 6)`), attenuate the reconstructed source contributions at the final query position using the fixed dose grid:
$$\beta \in \{1.0, 0.75, 0.5, 0.25, 0.0\}$$

#### Modes evaluated:
- `identity`: attenuate the identity-source vector $\Delta v_{\text{id}} = -(1 - \beta) v_{\text{id}}$.
- `evidence`: norm-matched attenuation along the evidence-source direction.
- `random_subset`: norm-matched attenuation along a deterministic random source-token subset direction.
- `whole_head`: unselective attenuation of the full head output (upper-bound side-effect control).

---

## Primary Outcomes & Estimands

1. **Clean margin**: $m_{\text{clean}} = \text{logit}(\text{Buy}) - \text{logit}(\text{Sell})$ on the clean prompt at `DECISION_PREFIX`.
2. **Intervened margin**: $m(\alpha)$ under suppression (or $m(\beta)$ under attenuation).
3. **Anonymous baseline margin**: $m_{\text{anon}}$ on the prompt with name/ticker replaced by `[ANON]` / `[Anonymous Company]`.
4. **Anonymous progress**:
   $$A_p(\alpha) = \frac{(m(\alpha) - m_{\text{clean}}) \cdot g}{g^2 + \varepsilon}, \quad g = m_{\text{anon}} - m_{\text{clean}}$$
5. **Decision flip**: boolean flag indicating whether the Buy/Sell sign flipped relative to clean.
6. **DLA mediation deltas**: change in identity-, evidence-, and instruction-sourced direct logit attribution for each selected head:
   $$\Delta D_{e,h}^{\text{source}}(\alpha) = D_{e,h}^{\text{source}}(\alpha) - D_{e,h}^{\text{source}}(1.0)$$
7. **Cross-ticker specificity contrast**:
   $$\Delta A_p^{\text{specificity}}(\alpha) = A_p^{\text{FTNT}}(\alpha) - \frac{1}{|\mathcal{P}|} \sum_{P \in \mathcal{P}} A_p^P(\alpha)$$
   where $\mathcal{P} = \{\text{ADI}, \text{MU}\}$ is the same-collision peer group.

---

## Scientific Interpretation Gates (Discovery Characterization)

Because E3 V1 is a discovery run, it reports descriptive curves and contrasts without an automated pass/fail gate. However, we pre-register the following interpretation criteria:

- **Evidence for FTNT-specific identity cell**:
  - $A_p^{\text{FTNT}}(-3.0) > 0$ and exceeds both within-ticker controls (`wrong_entity`, `matched_random`).
  - $A_p^{\text{FTNT}}(-3.0) > A_p^{\text{ADI}}(-3.0)$ and $A_p^{\text{FTNT}}(-3.0) > A_p^{\text{MU}}(-3.0)$ by at least $0.10$.
  - Evidence-sourced DLA is preserved ($|\Delta D_{\text{evidence}}| / |D_{\text{evidence}}| < 0.20$).

- **Evidence for generic entity-slot neuron**:
  - $A_p^{\text{FTNT}}(-3.0) \approx A_p^{\text{ADI}}(-3.0) \approx A_p^{\text{MU}}(-3.0) > 0$.
  - Suppressing `(0, 104)` moves all collision tickers toward their respective Anonymous baselines.

- **Evidence for inert / noisy neuron**:
  - $|A_p(-3.0)| \approx 0$ across all tickers, comparable to `matched_random`.

---

## Compact Output Artifacts

Following repository safety rules:
- No raw activations, residuals, attention weights, KV caches, or full-vocabulary logits are persisted.
- Compact outputs:
  - `e3/suppression.jsonl`: per-prompt compact records (ticker, prompt_id, phase, scope, dose, margin, clean_margin, anonymous_margin, flip, contributions, controls, provenance).
  - `e3/downstream.jsonl`: downstream attenuation compact records.
  - `analyze/summary.json`: aggregated group statistics (mean margin, anonymous progress, flip counts, mediation deltas, cross-ticker contrasts).
  - `manifest.json`: run manifest with input/output hashes, stage counts, and completion status.
