"""Train-only residual directions for binary-association steering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from llm_bias.counterfactual_patching.interventions import record_residuals


@dataclass(frozen=True)
class DirectionArtifact:
    """A compact, reproducible aggregate intervention direction."""

    layer: int
    dimension: int
    norm: float
    positive_label: str
    negative_label: str
    high_quantile: float
    low_quantile: float
    source_record_ids: list[str]
    source_hash: str
    model: str | None = None
    tokenizer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _span_mean(residual: torch.Tensor, spans: Sequence[tuple[int, int]]) -> torch.Tensor:
    values = [residual[:, start:end, :].mean(dim=1) for start, end in spans]
    if not values:
        raise ValueError("at least one span is required")
    return torch.stack(values, dim=0).mean(dim=0)


def _record_spans(record: dict[str, Any]) -> list[tuple[int, int]]:
    spans = record.get("entity_spans") or record.get("spans")
    if not spans:
        raise ValueError("direction record is missing entity spans")
    result = []
    for span in spans:
        if isinstance(span, dict):
            result.append((int(span["token_start"]), int(span["token_end"])))
        else:
            result.append((int(span[0]), int(span[1])))
    return result


def _record_input_ids(record: dict[str, Any], device: torch.device) -> torch.Tensor:
    values = record.get("input_ids")
    if values is None:
        raise ValueError("direction record is missing input_ids")
    tensor = torch.as_tensor(values, dtype=torch.long)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def fit_direction(
    model: Any,
    records: Iterable[dict[str, Any]],
    *,
    layer: int,
    device: torch.device | str,
    high_quantile: float = 0.75,
    low_quantile: float = 0.25,
    positive_label: str = "mother-associated",
    negative_label: str = "father-associated",
    model_name: str | None = None,
    tokenizer_name: str | None = None,
) -> tuple[torch.Tensor, DirectionArtifact]:
    """Fit a normalized mean-difference direction using only supplied records.

    Records must already belong to the training split.  The function does not
    inspect or retain example-level residuals after computing the aggregate.
    """
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1")
    materialized = list(records)
    if len(materialized) < 2:
        raise ValueError("at least two training records are required")
    device = torch.device(device)
    margins = torch.tensor(
        [float(record["margin"]) for record in materialized], dtype=torch.float32
    )
    high = torch.quantile(margins, high_quantile)
    low = torch.quantile(margins, low_quantile)
    positive: list[torch.Tensor] = []
    negative: list[torch.Tensor] = []
    all_ids = [str(record.get("record_id", record.get("career_id", index))) for index, record in enumerate(materialized)]
    ids: list[str] = []
    for record, margin in zip(materialized, margins, strict=True):
        residuals = record_residuals(model, _record_input_ids(record, device), [layer])
        vector = _span_mean(residuals[layer], _record_spans(record)).squeeze(0).detach().cpu()
        record_id = str(record.get("record_id", record.get("career_id", len(ids))))
        if margin >= high:
            positive.append(vector)
            ids.append(record_id)
        elif margin <= low:
            negative.append(vector)
            ids.append(record_id)
    if not positive or not negative:
        raise ValueError("quantile groups did not contain both positive and negative records")
    direction = torch.stack(positive).mean(dim=0) - torch.stack(negative).mean(dim=0)
    norm = float(direction.norm())
    if norm <= 1e-8:
        raise ValueError("training direction has zero norm")
    direction = direction / norm
    source_hash = hashlib.sha256(
        "\n".join(sorted(all_ids)).encode("utf-8")
    ).hexdigest()
    artifact = DirectionArtifact(
        layer=layer,
        dimension=int(direction.numel()),
        norm=1.0,
        positive_label=positive_label,
        negative_label=negative_label,
        high_quantile=high_quantile,
        low_quantile=low_quantile,
        source_record_ids=sorted(set(all_ids)),
        source_hash=source_hash,
        model=model_name,
        tokenizer=tokenizer_name,
    )
    return direction, artifact


def save_direction(
    path: str | Path,
    direction: torch.Tensor,
    metadata: DirectionArtifact,
) -> None:
    """Persist only an aggregate vector and compact metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    vector = direction.detach().float().cpu().flatten()
    if vector.numel() != metadata.dimension:
        raise ValueError("direction dimension does not match metadata")
    torch.save({"direction": vector, "metadata": metadata.to_dict()}, output)
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_direction(path: str | Path) -> tuple[torch.Tensor, DirectionArtifact]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "direction" not in payload or "metadata" not in payload:
        raise ValueError("direction artifact must contain direction and metadata")
    direction = torch.as_tensor(payload["direction"], dtype=torch.float32).flatten()
    metadata = DirectionArtifact(**payload["metadata"])
    if direction.numel() != metadata.dimension:
        raise ValueError("direction artifact dimension mismatch")
    return direction, metadata


def norm_matched_random_direction(direction: torch.Tensor, *, seed: int = 0) -> torch.Tensor:
    """Return a deterministic random direction with equal norm."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch.randn(direction.shape, generator=generator, dtype=direction.dtype)
    random = random / random.norm().clamp_min(1e-8)
    return random * direction.norm()


def permuted_direction(direction: torch.Tensor) -> torch.Tensor:
    """Return a deterministic coordinate permutation control."""
    if direction.numel() < 2:
        raise ValueError("direction must have at least two dimensions")
    indices = torch.arange(direction.numel() - 1, -1, -1)
    return direction.flatten()[indices].reshape_as(direction)
