"""Artifact-only contract for synthetic entity-bias visualization."""

from __future__ import annotations

ENTITY_POOL_FIELDS = (
    "ticker", "company_name", "latest_year", "years", "memberships",
    "membership_years", "sectors", "familiarity_tier", "source_row_count",
    "anomalies", "split",
)
BASELINE_FIELDS = (
    "template", "entity", "probabilities", "expected_score", "entropy_nats",
    "effective_temperature",
)
RESULT_FIELDS = (
    "ticker", "company_name", "template", "split", "familiarity_tier",
    "entity_probabilities", "baseline_probabilities", "entity_expected_score",
    "baseline_expected_score", "entity_entropy_nats", "baseline_entropy_nats",
    "entity_effective_temperature", "baseline_effective_temperature",
    "delta_expected_score", "entity_span_start", "entity_span_end",
    "answer_position",
)
LOCALIZATION_FIELDS = (
    "layer", "template", "mean_cosine", "pearson_r", "spearman_r",
    "linear_r2", "n_train", "n_eval", "q25", "q75", "n_high", "n_low",
    "high_ids_sha256", "low_ids_sha256", "fit_split", "direction_sha256",
    "statistic_flag",
)

TEMPLATE_ORDER = ("negative", "positive", "neutral")
TIER_ORDER = ("S&P 500", "Russell 1000", "Russell 2000")
REQUIRED_STAGES = ("preflight", "baseline", "metric", "localization")
REQUIRED_OUTPUTS = {
    "config": ("config.json", "preflight", None),
    "tokenization_validation": ("tokenization_validation.json", "preflight", None),
    "entity_pool": ("entity_pool.csv", "preflight", ENTITY_POOL_FIELDS),
    "no_entity_baselines": ("no_entity_baselines.csv", "baseline", BASELINE_FIELDS),
    "raw_entity_template_results": (
        "raw_entity_template_results.csv", "metric", RESULT_FIELDS
    ),
    "layer_template_localization": (
        "layer_template_localization.csv", "localization", LOCALIZATION_FIELDS
    ),
}
VISUALIZATION_SCHEMA_VERSION = 4
FIGURE_IDS = (
    "entity_effect_distribution",
    "entity_effect_tail_diagnostics",
    "baseline_entity_movement",
    "temperature_null_diagnostics",
    "entity_effect_by_tier",
    "template_relationships",
    "localization_profiles",
    "sector_effects",
)
FIGURE_FORMATS = ("png", "svg", "pdf")
SUMMARY_FILES = {
    "template": "template_summary.csv",
    "tier": "familiarity_tier_summary.csv",
    "sector": "sector_summary.csv",
    "ticker": "ticker_template_effects.csv",
    "localization": "localization_summary.csv",
    "tail_diagnostics": "entity_effect_tail_diagnostics.csv",
    "baseline_movement": "baseline_entity_movement.csv",
    "temperature_null": "temperature_null_diagnostics.csv",
    "localization_transitions": "localization_transition_diagnostics.csv",
}
