"""Fixed, historical-data-informed localization candidates, never effect selection.

The actual layer count must come from the loaded model's ordered layers. This
CPU record authenticates that supplied count against the frozen model registry,
not the checkpoint itself. No outcomes, tensors, cohort or sampling inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from .artifact_paths import sha256_json

HISTORICAL_SOURCE = 'docs/concept-cone-steering/c2-v3-steering-prompt/status.md'
HISTORICAL_SOURCE_SHA256 = '8a7762e97b934562167c6450bfdf441dca43c6815dc94bd97c20b2b22669b1b7'
SPANS = ('entity', 'evidence1', 'evidence2', 'instruction')
DEVELOPMENT_ROLES = ('fit', 'validation')
FULL_PAIR_COUNT = 4024
DEVELOPMENT_PAIR_COUNT = 3016
_MODELS = MappingProxyType({
    'qwen3.5-4b': (32, 15, 16),
    'glm4-9b-0414': (40, 20, 19),
    'gemma4-12b-it': (48, 27, 27),
    'gpt-oss-20b': (24, 8, 14),
})
_LIMITATIONS = (
    'historical_teacher_forced_clean_path_margin_not_generated_decision_proof',
    'historical_fixed_prefix_secondary_may_be_off_path',
    'historical_steering_prompt_construction_cross_company_not_current_site_proof',
    'qwen_glm_R7_opposite_clean_denominator_zero_not_zero_percent',
    'gpt_v3_band_excludes_v2_injection_layer',
    'historical_data_informed_design_not_untouched_preregistration',
)


def _candidate_layers(count: int, peak: int, injection: int) -> tuple[int, ...]:
    """Clip historical neighbors before deduplicating with independent anchors."""
    return tuple(sorted({layer for layer in
                         (peak - 1, peak, peak + 1, injection,
                          0, (count - 1) // 8, (count - 1) // 2, count - 1)
                         if 0 <= layer < count}))


@dataclass(frozen=True, slots=True)
class LocalizationCandidatePanel:
    """Closed immutable panel identity. No alternate layers or source accepted."""

    model_slug: str
    actual_layer_count: int

    def __post_init__(self) -> None:
        if type(self.model_slug) is not str or self.model_slug not in _MODELS:
            raise ValueError('expected a canonical registered model slug')
        count = self.actual_layer_count
        if type(count) is not int or count <= 0:
            raise ValueError('actual layer count must be a genuine positive integer')
        if count != _MODELS[self.model_slug][0]:
            raise ValueError('actual layer count differs from frozen expected count')
        source = Path(__file__).resolve().parents[2] / HISTORICAL_SOURCE
        if sha256(source.read_bytes()).hexdigest() != HISTORICAL_SOURCE_SHA256:
            raise ValueError('historical source differs from frozen SHA-256')

    @property
    def layers(self) -> tuple[int, ...]:
        count, peak, injection = _MODELS[self.model_slug]
        return _candidate_layers(count, peak, injection)

    @property
    def development_cell_count(self) -> int:
        return DEVELOPMENT_PAIR_COUNT * len(SPANS) * len(self.layers)

    def to_dict(self) -> dict:
        """Fresh canonical compact provenance, without observed effects."""
        count, peak, injection = _MODELS[self.model_slug]
        return dict(
            kind='stance_localization_candidate_panel_v2',
            model_slug=self.model_slug, actual_layer_count=self.actual_layer_count,
            expected_layer_count=count, layers=list(self.layers),
            rule='v3_steer_suffix_peak_plus_minus_1_union_v2_injection_union_0_early_mid_last',
            index_base=0, early_divisor=8, mid_divisor=2,
            historical_peak=peak, historical_injection_layer=injection,
            historical_source=HISTORICAL_SOURCE,
            historical_source_sha256=HISTORICAL_SOURCE_SHA256,
            historical_limitations=list(_LIMITATIONS),
            site='post', spans=list(SPANS), selector='full',
            development_roles=list(DEVELOPMENT_ROLES), full_pair_count=FULL_PAIR_COUNT,
            development_pair_count=DEVELOPMENT_PAIR_COUNT,
            development_cell_count=self.development_cell_count,
            objective='within_fixed_panel_candidate_comparison_not_global_peak_or_band',
        )

    @property
    def panel_sha256(self) -> str:
        return sha256_json(self.to_dict())


def build_localization_candidate_panel(*, model_slug: str,
                                       actual_layer_count: int) -> LocalizationCandidatePanel:
    """Require canonical model identity and observed count, with no CLI defaults."""
    return LocalizationCandidatePanel(model_slug, actual_layer_count)
