"""Validated compact schemas for J-space intervention configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
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
class InterventionConfig:
    source: PrototypeSpec
    target: PrototypeSpec
    layers: tuple[int, ...]
    alphas: tuple[float, ...]
    coordinate_modes: tuple[str, ...] = ("swap",)
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
            alphas=tuple(float(alpha) for alpha in value["alphas"]),
            coordinate_modes=tuple(value.get("coordinate_modes", ("swap",))),
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
        if not result.alphas or 0.0 not in result.alphas:
            raise ValueError("alphas must include the no-op dose 0")
        valid_modes = {"swap", "source_ablation", "target_addition"}
        if not result.coordinate_modes or not set(result.coordinate_modes) <= valid_modes:
            raise ValueError("coordinate_modes contains an unsupported mode")
        if len(result.coordinate_modes) != len(set(result.coordinate_modes)):
            raise ValueError("coordinate_modes must be unique")
        if result.top_positions <= 0:
            raise ValueError("top_positions must be positive")
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


__all__ = ["ConceptToken", "InterventionConfig", "PrototypeSpec"]
