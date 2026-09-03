# Entity Cell Localization: Proposal V2 (E1 surface-varying localization)

**Document status:** V2 protocol frozen 2026-09-02 (commit `91bd657`) before
any V2 model run; V2 discovery run complete — see [report-v2](report-v2.md).
Calibration and held-out test not run.

**Scope:** this document is a version of the **E1 phase only**. It changes the
localization prompt family, the control family, and the success gates.
Everything else — input population, E2 (downstream component attribution),
E3 (upstream/downstream suppression), the confirmation freeze design, claim
boundaries, and milestones — is unchanged and remains defined in
[proposal-v1](proposal-v1.md). Version index: [README](README.md).

## Motivation (V1 discovery finding)

The V1 discovery run (`entity-cell-e1-discovery-v1`) localized 31/35
Technology tickers to the same top-1 neuron (L0, N4485) and 4 to a second
shared neuron (L0, N5101); all three surface-form controls showed top-5
overlap 5/5 with the localization top-5; zero tickers passed the amnesia
eligibility gate. The mechanism is a property of the V1 prompt family: all
twelve variants share the identical three-line header (`Stock Ticker: [X]` /
`Stock Name: [Y]` / `--- Evidence ---`), so a neuron that reacts to the header
template — not to the company — has near-zero cross-variant standard
deviation and therefore dominates the stability score
$S_{\ell j} = (\mathbb{E}_i z)^2/(\operatorname{Std}_i z + \varepsilon)$ for
every ticker. Barzilay et al. (2026) avoid this by localizing on prompts whose
entity surface context varies per prompt (`The <attribute> of <entity>`, a
100-attribute template list, K=2 prompts per entity), so template-reactive
neurons are not stable across variants. V2 changes the localization prompt
family accordingly; V1 results stand as reported and are never re-labeled as
V2 results.

## V2 localization prompt family (natural-sentence frames)

For each company with name $N$ and ticker $X$, render twelve frozen sentence
frames; eight form the localization set and four the unseen held set. The
company name appears once per frame as plain prose (no brackets, no fixed
label), and the entity token position is the last content token of the name
span in that frame, using the shared character-to-token span helper:

| id | set | frame |
|---|---|---|
| F0 | localization | `The headquarters of {name} is located in` |
| F1 | localization | `{name} is a technology company.` |
| F2 | localization | `The stock ticker of {name} is` |
| F3 | localization | `{name} was founded in` |
| F4 | localization | `The main product of {name} is` |
| F5 | localization | `Investors often describe {name} as` |
| F6 | localization | `{name} operates in the` |
| F7 | localization | `The annual report of {name} states that` |
| H0 | held | `The CEO of {name} is` |
| H1 | held | `{name} is headquartered in` |
| H2 | held | `The market value of {name} reached` |
| H3 | held | `{name} competes with other` |

Frames are prompt-only: they are never generated or answered, and no frame
asserts a per-company fact after the name span.

## Comparison and template controls

- **Header-family comparison control:** the frozen V1 header family (twelve
  header-prefix variants) is re-run on the same tickers and reported for
  comparison only; it is not used for V2 ranking. It must reproduce the V1
  surface-dominated ranking, and V2 reports how many V2 candidates overlap it.
- **Template-only control (new):** one frozen prompt with the V1 header
  structure and neutral content —
  `Stock Ticker: [NEUT]` / `Stock Name: [Neutral Entity, Inc.]` — whose top-5
  is the frozen template signature. A candidate that is top-5 in the template
  signature is template-reactive, not identity-selective.
- **Surface-form controls:** in the frame family (the V2 ranking family), the
  eight localization frames are re-rendered at run time with the company name
  span replaced: `anonymous_name_frames` (name → `Anonymous Company`) and
  `name_form_control_frames` (name → ROT13 of the name). In the header
  comparison family, the three frozen V1 controls (`anonymous_ticker`,
  `anonymous_name`, `name_form_control`) are reported for comparison only. A
  frame-family candidate is form-robust only if it is not top-5 in either
  frame-family control ranking; top-5 overlap with each control is reported
  descriptively.

## V2 gates

A V2 candidate is a trusted candidate entity cell only if all of the
following hold:

1. held-variant top-5 overlap $\geq 1$ (V1 rule);
2. form-robust: not top-5 in either frame-family surface control
   (`anonymous_name_frames` or `name_form_control_frames`);
3. template-robust (new): not top-5 in the template-only control signature;
4. amnesia endpoint gate (V1 rule): at $\alpha=-3$, $A_p(-3)>0$ and exceeds
   both control curves on at least two eligible prompts.

## Wrong-entity non-degeneracy (new rule)

V1 showed that when every ticker shares one top-1 neuron, the wrong-entity
control equals the target cell and the gate is uninformative. V2 fixes the
wrong entity deterministically (the next ticker alphabetically within the
same split) and uses its top-1 cell; if that cell equals the target cell,
descend to its top-2, and so on through top-5. If all five match, the
wrong-entity control is marked degenerate, the amnesia gate must be satisfied
against the matched-random control alone, and the ticker is flagged as
degraded-control in the summary.

## Unchanged from V1

The 399 generic baseline prompts (same file and identity), per-neuron
$(\mu_{\ell j}, \sigma_{\ell j})$ normalization, $\varepsilon = 10^{-6}$,
candidate layers L0–L5, top-5 retention, amnesia dose grid
$\alpha \in \{1,0,-1,-2,-3\}$, three financial prompts per ticker, and the
all-positions primary scope.

## Versioning boundary

V2 requires its own tokenizer-only preparation run (V2 frames, header-family
comparison control, template-only control; reusing the existing generic
baseline and financial prompts) and its own model runs. V1 and V2 cells are
never mixed in one analysis. The confirmation freeze (calibration/test) may
be built only from V2 discovery selections.
