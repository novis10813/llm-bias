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


__all__ = ["ConceptToken", "GainConfig", "InterventionConfig", "PrototypeSpec"]
