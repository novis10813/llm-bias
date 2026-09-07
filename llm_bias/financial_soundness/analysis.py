"""Research-specific paired response ranking and descriptive causal summaries."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any
import torch
from llm_bias.core.analysis.statistics import bootstrap_mean_ci

SCALE_FLOOR = 1e-6


def group_key(pair: dict) -> tuple[str, str, str]:
    return pair["family"], pair["concept"], pair["answer_mode"]


def response_stats(records: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    a = torch.stack([r[0] for r in records]).float()
    b = torch.stack([r[1] for r in records]).float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("non-finite response")
    delta = a - b
    scale = ((a.square() + b.square()).mean(0) / 2).sqrt()
    score = delta.mean(0) / scale.clamp_min(SCALE_FLOOR)
    return {"score": score, "scale": scale, "consistency": (delta.sign() == score.sign()).float().mean(0)}


def rank_candidates(responses: dict, top_k: int) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    stats = {key: response_stats(records) for key, records in responses.items()}
    groups = sorted({key[:3] for key in responses})
    result = []
    for group in groups:
        available = []
        for key, stat in stats.items():
            if key[:3] != group or key[3] != "discovery":
                continue
            layer = key[4]
            for neuron in torch.where(stat["scale"] > SCALE_FLOOR)[0].tolist():
                score = float(stat["score"][neuron])
                available.append((abs(score), layer, neuron))
        for rank, (_, layer, neuron) in enumerate(sorted(available, key=lambda x: (-x[0], x[1], x[2]))[:top_k], 1):
            splits = {}
            for split in ("discovery", "calibration", "held-out"):
                stat = stats[(*group, split, layer)]
                splits[split] = {"score": float(stat["score"][neuron]), "sign_consistency": float(stat["consistency"][neuron])}
            result.append(dict(family=group[0], concept=group[1], answer_mode=group[2],
                               layer=layer, neuron=neuron, rank=rank, splits=splits))
    return result


def summarize_behavior(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["concept"], row["answer_mode"], row["split"])].append(row)
    result = []
    for key, items in sorted(groups.items()):
        pair_rows = defaultdict(dict)
        for item in items:
            pair_rows[item["pair_id"]][item["member"]] = item
        clusters = defaultdict(list)
        for members in pair_rows.values():
            a, b = members["a"], members["b"]
            clusters[a["group_id"]].append(a["margin"] - b["margin"])
        correct = [r["correct"] for r in items if r["correct"] is not None]
        result.append(dict(family=key[0], concept=key[1], answer_mode=key[2], split=key[3],
                           pair_difference=bootstrap_mean_ci([mean(v) for v in clusters.values()], n_resamples=1000),
                           accuracy=mean(correct) if correct else None))
    return result


def summarize_effects(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("layer", "neuron", "family", "answer_mode", "split", "condition", "scale"))
        groups[key].append(row)
    result = []
    for key, items in groups.items():
        by_pair = defaultdict(dict)
        for row in items:
            by_pair[row["pair_id"]][row["member"]] = row
        clusters = defaultdict(lambda: defaultdict(list))
        for pair in by_pair.values():
            if set(pair) != {"a", "b"}:
                raise ValueError("incomplete intervention pair")
            a, b = pair["a"], pair["b"]
            da, db = a["margin"] - a["clean_margin"], b["margin"] - b["clean_margin"]
            values = clusters[a["group_id"]]
            values["discrimination_delta"].append(da - db)
            values["answer_shift"].append((da + db) / 2)
            values["correct_margin_delta"].append((a["correct_margin_delta"] + b["correct_margin_delta"]) / 2)
        item = dict(zip(("layer", "neuron", "family", "answer_mode", "split", "condition", "scale"), key))
        item["statistics"] = {name: bootstrap_mean_ci([mean(c[name]) for c in clusters.values()], n_resamples=1000)
                              for name in ("discrimination_delta", "answer_shift", "correct_margin_delta")}
        result.append(item)
    return result
