"""Validated compact schemas for J-space intervention configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class ConceptToken:
    token: str
    token_id: int
    weight: float
    selection_score: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConceptToken":
        result = cls(
            token=str(value["token"]),
            token_id=int(value["token_id"]),
            weight=float(value["weight"]),
            selection_score=(
                float(value["selection_score"])
                if value.get("selection_score") is not None
                else None
            ),
        )
        if not result.token or result.token_id < 0 or result.weight < 0:
            raise ValueError("invalid concept token")
        return result


@dataclass(frozen=True)
class PrototypeSpec:
    name: str
    sector: str
    score_type: str
    tokens: tuple[ConceptToken, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PrototypeSpec":
        result = cls(
            name=str(value["name"]),
            sector=str(value["sector"]),
            score_type=str(value["score_type"]),
            tokens=tuple(ConceptToken.from_dict(row) for row in value["tokens"]),
        )
        if not result.name or not result.sector or not result.tokens:
            raise ValueError("prototype name, sector, and tokens are required")
        ids = [token.token_id for token in result.tokens]
        if len(ids) != len(set(ids)):
            raise ValueError("prototype token IDs must be unique")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GainConfig:
    prototype: PrototypeSpec
    layers: tuple[int, ...]
    gains: tuple[float, ...]
    position_controls: tuple[str, ...] = ("evidence",)
    direction_controls: tuple[str, ...] = ("prototype",)
    top_positions: int = 3
    loading_threshold: float = 0.0
    decision_prefix: str = '{\n  "decision": "'
    positive_candidate: str = "buy"
    negative_candidate: str = "sell"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GainConfig":
        result = cls(
            prototype=PrototypeSpec.from_dict(value["prototype"]),
            layers=tuple(int(layer) for layer in value["layers"]),
            gains=tuple(float(gain) for gain in value["gains"]),
            position_controls=tuple(value.get("position_controls", ("evidence",))),
            direction_controls=tuple(value.get("direction_controls", ("prototype",))),
            top_positions=int(value.get("top_positions", 3)),
            loading_threshold=float(value.get("loading_threshold", 0.0)),
            decision_prefix=str(value.get("decision_prefix", '{\n  "decision": "')),
            positive_candidate=str(value.get("positive_candidate", "buy")),
            negative_candidate=str(value.get("negative_candidate", "sell")),
        )
        if not result.layers or len(result.layers) != len(set(result.layers)):
            raise ValueError("layers must be non-empty and unique")
        if not result.gains or 1.0 not in result.gains:
            raise ValueError("gains must include the no-op gain 1")
        if any(not math.isfinite(gain) or gain < 0.0 for gain in result.gains):
            raise ValueError("gains must be finite and non-negative")
        if result.top_positions <= 0:
            raise ValueError("top_positions must be positive")
        if not math.isfinite(result.loading_threshold):
            raise ValueError("loading_threshold must be finite")
        if not set(result.position_controls) <= {
            "evidence", "shuffled_evidence", "final_position"
        }:
            raise ValueError("unsupported position control")
        if not set(result.direction_controls) <= {"prototype", "matched_random"}:
            raise ValueError("unsupported direction control")
        answer_terms = {
            result.positive_candidate.strip().lower(),
            result.negative_candidate.strip().lower(),
        }
        prototype_terms = {
            token.token.strip().lower() for token in result.prototype.tokens
        }
        overlap = answer_terms & prototype_terms
        if overlap:
            raise ValueError(
                f"answer candidates cannot be prototype concepts: {sorted(overlap)}"
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InterventionConfig:
    source: PrototypeSpec
    target: PrototypeSpec
    layers: tuple[int, ...]
    swap_fractions: tuple[float, ...]
    coordinate_modes: tuple[str, ...] = ("swap",)
    position_controls: tuple[str, ...] = ("evidence",)
    direction_controls: tuple[str, ...] = ("prototype",)
    top_positions: int = 3
    loading_threshold: float = 0.0
    decision_prefix: str = '{\n  "decision": "'
    positive_candidate: str = "buy"
    negative_candidate: str = "sell"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InterventionConfig":
        result = cls(
            source=PrototypeSpec.from_dict(value["source"]),
            target=PrototypeSpec.from_dict(value["target"]),
            layers=tuple(int(layer) for layer in value["layers"]),
            swap_fractions=tuple(
                float(fraction)
                for fraction in value.get("swap_fractions", value.get("alphas", ()))
            ),
            coordinate_modes=tuple(
                {
                    "source_ablation": "source_removal_component",
                    "target_addition": "target_installation_component",
                }.get(mode, mode)
                for mode in value.get("coordinate_modes", ("swap",))
            ),
            position_controls=tuple(value.get("position_controls", ("evidence",))),
            direction_controls=tuple(value.get("direction_controls", ("prototype",))),
            top_positions=int(value.get("top_positions", 3)),
            loading_threshold=float(value.get("loading_threshold", 0.0)),
            decision_prefix=str(value.get("decision_prefix", '{\n  "decision": "')),
            positive_candidate=str(value.get("positive_candidate", "buy")),
            negative_candidate=str(value.get("negative_candidate", "sell")),
        )
        if result.source.sector == result.target.sector:
            raise ValueError("source and target sectors must differ")
        if not result.layers or len(result.layers) != len(set(result.layers)):
            raise ValueError("layers must be non-empty and unique")
        if not result.swap_fractions or 0.0 not in result.swap_fractions:
            raise ValueError("swap_fractions must include the no-op fraction 0")
        if any(
            not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0
            for fraction in result.swap_fractions
        ):
            raise ValueError("swap_fractions must be finite and remain within [0, 1]")
        valid_modes = {
            "swap",
            "source_removal_component",
            "target_installation_component",
        }
        if not result.coordinate_modes or not set(result.coordinate_modes) <= valid_modes:
            raise ValueError("coordinate_modes contains an unsupported mode")
        if len(result.coordinate_modes) != len(set(result.coordinate_modes)):
            raise ValueError("coordinate_modes must be unique")
        if result.top_positions <= 0:
            raise ValueError("top_positions must be positive")
        if not math.isfinite(result.loading_threshold):
            raise ValueError("loading_threshold must be finite")
        if not set(result.position_controls) <= {
            "evidence", "shuffled_evidence", "final_position"
        }:
            raise ValueError("unsupported position control")
        if not set(result.direction_controls) <= {"prototype", "matched_random"}:
            raise ValueError("unsupported direction control")
        answer_terms = {
            result.positive_candidate.strip().lower(),
            result.negative_candidate.strip().lower(),
        }
        prototype_terms = {
            token.token.strip().lower()
            for prototype in (result.source, result.target)
            for token in prototype.tokens
        }
        overlap = answer_terms & prototype_terms
        if overlap:
            raise ValueError(
                f"answer candidates cannot be prototype concepts: {sorted(overlap)}"
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenScreenCandidate:
    """One frozen valence candidate copied into a token screen config.

    ``representation_side`` is provenance only: the screen never multiplies
    an intervention by the side sign.
    """

    token: str
    token_id: int
    representation_side: str
    mean_positive: float
    mean_negative: float
    band_probability_diff: float
    band_smoothed_log_ratio: float
    band_js_contribution: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TokenScreenCandidate":
        side = value.get("representation_side", value.get("side"))
        result = cls(
            token=str(value["token"]),
            token_id=int(value["token_id"]),
            representation_side=str(side),
            mean_positive=float(value["mean_positive"]),
            mean_negative=float(value["mean_negative"]),
            band_probability_diff=float(value["band_probability_diff"]),
            band_smoothed_log_ratio=float(value["band_smoothed_log_ratio"]),
            band_js_contribution=float(value["band_js_contribution"]),
        )
        if not result.token or result.token_id < 0:
            raise ValueError("invalid token screen candidate")
        if result.representation_side not in {"positive", "negative"}:
            raise ValueError("candidate representation side must be positive or negative")
        scores = (
            result.mean_positive,
            result.mean_negative,
            result.band_probability_diff,
            result.band_smoothed_log_ratio,
            result.band_js_contribution,
        )
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("candidate readout scores must be finite")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenScreenConfig:
    candidates: tuple[TokenScreenCandidate, ...]
    model: str
    source_sector: str
    layers: tuple[int, ...]
    alphas: tuple[float, ...] = (-1.0, 0.0, 1.0)
    top_positions: int = 3
    loading_threshold: float = 0.0
    controls: tuple[str, ...] = ("token", "matched_random")
    decision_prefix: str = '{\n  "decision": "'
    positive_candidate: str = "buy"
    negative_candidate: str = "sell"
    outcome_scoring: str = "single_token_fp32_final_norm_unembedding"
    candidate_artifact_path: str = ""
    candidate_artifact_sha256: str = ""
    split_manifest_sha256: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TokenScreenConfig":
        result = cls(
            candidates=tuple(
                TokenScreenCandidate.from_dict(row) for row in value["candidates"]
            ),
            model=str(value.get("model", "")),
            source_sector=str(value["source_sector"]),
            layers=tuple(int(layer) for layer in value["layers"]),
            alphas=tuple(float(alpha) for alpha in value.get("alphas", (-1.0, 0.0, 1.0))),
            top_positions=int(value.get("top_positions", 3)),
            loading_threshold=float(value.get("loading_threshold", 0.0)),
            controls=tuple(value.get("controls", ("token", "matched_random"))),
            decision_prefix=str(value.get('decision_prefix', '{\n  "decision": "')),
            positive_candidate=str(value.get("positive_candidate", "buy")),
            negative_candidate=str(value.get("negative_candidate", "sell")),
            outcome_scoring=str(
                value.get("outcome_scoring", "single_token_fp32_final_norm_unembedding")
            ),
            candidate_artifact_path=str(value.get("candidate_artifact_path", "")),
            candidate_artifact_sha256=str(value.get("candidate_artifact_sha256", "")),
            split_manifest_sha256=str(value.get("split_manifest_sha256", "")),
        )
        if not result.model:
            raise ValueError("model is required")
        if not result.source_sector:
            raise ValueError("source_sector is required")
        if not result.candidates:
            raise ValueError("token screen requires at least one candidate")
        token_ids = [candidate.token_id for candidate in result.candidates]
        if len(token_ids) != len(set(token_ids)):
            raise ValueError("candidate token IDs must be unique")
        if not result.layers or len(result.layers) != len(set(result.layers)):
            raise ValueError("layers must be non-empty and unique")
        doses = sorted(result.alphas)
        if (
            len(doses) != 3
            or len(set(doses)) != 3
            or doses[1] != 0.0
            or doses[0] != -doses[2]
            or doses[2] <= 0.0
            or any(not math.isfinite(dose) for dose in doses)
        ):
            raise ValueError(
                "alphas must be exactly one symmetric nonzero dose pair plus 0 "
                "(e.g. -1, 0, 1)"
            )
        if result.top_positions <= 0:
            raise ValueError("top_positions must be positive")
        if not math.isfinite(result.loading_threshold):
            raise ValueError("loading_threshold must be finite")
        if not result.controls or len(result.controls) != len(set(result.controls)):
            raise ValueError("controls must be a non-empty unique set")
        if not set(result.controls) <= {"token", "matched_random"}:
            raise ValueError("unsupported token screen control")
        answer_terms = {
            result.positive_candidate.strip().lower(),
            result.negative_candidate.strip().lower(),
        }
        if not answer_terms or answer_terms == {""}:
            raise ValueError("answer candidates must be non-empty")
        if result.outcome_scoring != "single_token_fp32_final_norm_unembedding":
            raise ValueError("unsupported token screen outcome scorer")
        candidate_terms = {
            candidate.token.strip().lower() for candidate in result.candidates
        }
        overlap = answer_terms & candidate_terms
        if overlap:
            raise ValueError(
                f"answer candidates cannot be screen candidates: {sorted(overlap)}"
            )
        for label, sha in (
            ("candidate_artifact_sha256", result.candidate_artifact_sha256),
            ("split_manifest_sha256", result.split_manifest_sha256),
        ):
            if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
                raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
        if not result.candidate_artifact_path:
            raise ValueError("candidate_artifact_path is required")
        return result

    @property
    def positive_dose(self) -> float:
        """The positive dose ``a`` of the configured ``(-a, 0, a)"` pair."""
        return max(alpha for alpha in self.alphas if alpha > 0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutcomeFlipConfig:
    """Frozen V2 outcome-conditioned decision-flip parameters (Draft 1).

    The direction source is the outcome-gradient axis fitted from discovery
    prompts; it is never part of this config.  Calibration/test runs bind it
    through SHA-verified direction-identity and selection artifacts.
    """

    model: str
    source_sector: str
    fitted_layers: tuple[int, ...]
    candidate_bands: tuple[tuple[int, int], ...]
    position_rules: tuple[str, ...]
    dose_grid: tuple[float, ...]
    split_manifest_sha256: str
    safety_bound: float = 0.50
    scale_floor: float = 1.0
    tie_rule: str = "exclude_exact_zero"
    clean_margin_edges: tuple[float, ...] = (0.5, 1.5)
    min_flip_rate: float = 0.10
    parse_success_gate: float = 0.90
    agreement_gate: float = 0.70
    max_new_tokens: int = 256
    fitting_seed: int = 0
    bootstrap_seed: int = 0
    bootstrap_samples: int = 2000
    decision_prefix: str = '{\n  "decision": "'
    positive_candidate: str = "buy"
    negative_candidate: str = "sell"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OutcomeFlipConfig":
        result = cls(
            model=str(value.get("model", "")),
            source_sector=str(value["source_sector"]),
            fitted_layers=tuple(int(layer) for layer in value["fitted_layers"]),
            candidate_bands=tuple(
                (int(start), int(end))
                for start, end in value["candidate_bands"]
            ),
            position_rules=tuple(value["position_rules"]),
            dose_grid=tuple(float(dose) for dose in value["dose_grid"]),
            split_manifest_sha256=str(value.get("split_manifest_sha256", "")),
            safety_bound=float(value.get("safety_bound", 0.50)),
            scale_floor=float(value.get("scale_floor", 1.0)),
            tie_rule=str(value.get("tie_rule", "exclude_exact_zero")),
            clean_margin_edges=tuple(
                float(edge) for edge in value.get("clean_margin_edges", (0.5, 1.5))
            ),
            min_flip_rate=float(value.get("min_flip_rate", 0.10)),
            parse_success_gate=float(value.get("parse_success_gate", 0.90)),
            agreement_gate=float(value.get("agreement_gate", 0.70)),
            max_new_tokens=int(value.get("max_new_tokens", 256)),
            fitting_seed=int(value.get("fitting_seed", 0)),
            bootstrap_seed=int(value.get("bootstrap_seed", 0)),
            bootstrap_samples=int(value.get("bootstrap_samples", 2000)),
            decision_prefix=str(value.get('decision_prefix', '{\n  "decision": "')),
            positive_candidate=str(value.get("positive_candidate", "buy")),
            negative_candidate=str(value.get("negative_candidate", "sell")),
        )
        if not result.model:
            raise ValueError("model is required")
        if not result.source_sector:
            raise ValueError("source_sector is required")
        if not result.fitted_layers or len(result.fitted_layers) != len(set(result.fitted_layers)):
            raise ValueError("fitted_layers must be non-empty and unique")
        if any(layer < 0 for layer in result.fitted_layers):
            raise ValueError("fitted_layers must be non-negative")
        if not result.candidate_bands:
            raise ValueError("candidate_bands must be non-empty")
        fitted = set(result.fitted_layers)
        for start, end in result.candidate_bands:
            if start > end:
                raise ValueError("candidate band start must not exceed its end")
            if not set(range(start, end + 1)) <= fitted:
                raise ValueError("candidate band layers must be a subset of fitted_layers")
        if len({band for band in result.candidate_bands}) != len(result.candidate_bands):
            raise ValueError("candidate_bands must be unique")
        if not result.position_rules or len(result.position_rules) != len(set(result.position_rules)):
            raise ValueError("position_rules must be a non-empty unique set")
        if not set(result.position_rules) <= {"evidence_item_end", "evidence_span_all"}:
            raise ValueError("unsupported outcome flip position rule")
        if not result.dose_grid or len(result.dose_grid) != len(set(result.dose_grid)):
            raise ValueError("dose_grid must be a non-empty unique set")
        if any(
            not math.isfinite(dose) or dose <= 0.0 or dose > result.safety_bound
            for dose in result.dose_grid
        ):
            raise ValueError("dose_grid values must be finite, positive, and within the safety bound")
        if not 0.0 < result.safety_bound <= 1.0:
            raise ValueError("safety_bound must be within (0, 1]")
        if result.scale_floor <= 0.0 or not math.isfinite(result.scale_floor):
            raise ValueError("scale_floor must be finite and positive")
        if result.tie_rule != "exclude_exact_zero":
            raise ValueError("tie_rule must be the frozen 'exclude_exact_zero' rule")
        edges = list(result.clean_margin_edges)
        if any(not math.isfinite(edge) or edge <= 0.0 for edge in edges):
            raise ValueError("clean_margin_edges must be finite and positive")
        if edges != sorted(set(edges)):
            raise ValueError("clean_margin_edges must be strictly increasing")
        if not 0.0 < result.min_flip_rate <= 1.0:
            raise ValueError("min_flip_rate must be within (0, 1]")
        if not 0.0 < result.parse_success_gate <= 1.0:
            raise ValueError("parse_success_gate must be within (0, 1]")
        if not 0.0 < result.agreement_gate <= 1.0:
            raise ValueError("agreement_gate must be within (0, 1]")
        if result.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if result.bootstrap_samples < 2:
            raise ValueError("bootstrap_samples must be at least 2")
        if not {
            result.positive_candidate.strip().lower(),
            result.negative_candidate.strip().lower(),
        } - {""}:
            raise ValueError("answer candidates must be non-empty")
        if len(result.split_manifest_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in result.split_manifest_sha256
        ):
            raise ValueError("split_manifest_sha256 must be a lowercase SHA-256 hex digest")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "ConceptToken",
    "GainConfig",
    "InterventionConfig",
    "OutcomeFlipConfig",
    "PrototypeSpec",
    "TokenScreenCandidate",
    "TokenScreenConfig",
]
