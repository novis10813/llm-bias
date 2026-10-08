# Shared FactSet context defines the main evidence estimand

The user selected externally sourced **S&P 500 aggregate financial context** for the main experiment. All 503 ticker prompts receive identical two-slot evidence in each condition; this tests company identity under shared market context, not company-specific financial evidence. The population remains the exact 2024 CSV. Report date is 2024-11-15; do not describe this as a historical investment backtest or claim known model training cutoffs.

## Independent machine review checked the source and corrected the items

Source: FactSet, John Butters, *Earnings Insight*, 2024-11-15. Original URL: https://advantage.factset.com/hubfs/Website/Resources%20Section/Research%20Desk/Earnings%20Insight/EarningsInsight_111524.pdf.

Acquired file: untracked `data/concept-cone-steering/rebuild-v1/sources/factset_20241115.pdf`, 1,080,178 bytes, SHA-256 `677776cd01ddf02efcfb5fb4b437c336baa825f0dd41f3cdf9f3b1b829078486`. Local text extraction uses `pdftotext -layout`; PDF remains the authoritative bytes. Main agent checked headline/growth/guidance definitions against the extraction. Independent machine factual review verified source bytes, extraction, periods and denominators; its required N2-definition and page-reference corrections are incorporated below. This review does not supply blinded human reason labels. Source hash proves byte identity, not interpretation or item quality.

## Proposed four items retain qualifications and attribution

These are proposed attributed paraphrases, not claims every research company has these results. Polarity labels describe selected supportive/adverse factors; they do not establish objectively correct buy/sell labels or equal persuasive strength. Strength is provisionally `single_aggregate_factor`; no balanced-strength claim is made.

| ID / label | Proposed exact text | Source check |
|---|---|---|
| F24-P1 / + | As of November 15, 2024, FactSet reported a blended year-over-year earnings growth rate of 5.4% for the S&P 500 in Q3 2024; blended combines reported results with estimates for companies that had not yet reported. | PDF pp.1,5,9; 93% reported as of this issue; blended is not final realized growth. |
| F24-P2 / + | As of November 15, 2024, FactSet reported a blended year-over-year revenue growth rate of 5.5% for the S&P 500 in Q3 2024, compared with an estimated growth rate of 4.7% on September 30, 2024. | PDF pp.5,8,10; do not say every company increased revenue. |
| F24-N1 / - | As of November 15, 2024, FactSet reported an S&P 500 forward 12-month P/E ratio of 22.0, above its five-year average of 19.6 and ten-year average of 18.1. | PDF pp.1,6,12; elevated multiple is a valuation factor, not proof of overvaluation or future loss. |
| F24-N2 / - | As of November 15, 2024, FactSet reported that 54 of the 80 S&P 500 companies issuing EPS guidance for Q4 2024 had provided an EPS estimate, or the midpoint of an estimate range, below the mean analyst EPS estimate from the day before the guidance was issued; FactSet classified these as negative guidance. | PDF p.12; 54/80=67.5%, rounded 68% in report. Below-consensus guidance is not necessarily a downward revision to previous company guidance. |

Condition pairs: `++` P1/P2; `+-` P1/N1; `-+` N1/P1; `--` N1/N2. The mixed pair is an exact item/content swap. Trial `factset-20241115-a` is a single registered factor panel; more evidence trials, if added, require pre-run registration and company-first aggregation. Slot labels cannot expose polarity to the model.

## Pair compilation must retain the completed review provenance

An independent machine reviewer checked each number, its unit/period/denominator, report version/page, blended-vs-realized qualification, attribution, guidance definition and unsupported causal or investment interpretation. Reject a paraphrase adding conclusions absent from the source. Save exact item text/content hash, source path/hash, reviewer identity/type, review record and limitations; machine factual review is labeled machine review, not blinded human reason annotation.

Company-info and instruction remain fixed across conditions. A fixed instruction will explicitly say these are aggregate market facts, not company-specific results, and request an investment stance for the named stock using the provided shared context. No source item claims that market averages describe the named issuer. Company-specific evidence and external-source alternatives belong to later versions, not this primary table.
