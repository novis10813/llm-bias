"""Entity-cell factual amnesia probe (proposed; not a frozen protocol).

Closes the ROME/Barzilay verification loop that the original FTNT line skipped:
suppress the localized entity cell and measure whether the model's factual
recall about that entity collapses, with dose response and controls.

Calibrated on entity-cell-e1-hfm-discovery-v1 (AMZN, (L0, N1476), score
3386.7) and generalized via CLI for further HFM candidates (see
--entity/--cell/--wrong-cell/--cross). The wrong-entity control is any
other entity's cell; the matched-random neuron is deterministic from the
source run's generic baseline stats.

Prompt-driven, raw-text factual cloze frames (same framing as the factual
recall preflight; no chat template). Gold sequence = the model's own clean
greedy first-3-token continuation, pre-verified human-side to be the correct
fact; cross frames (other entities' HQ frame) verify the neuron is not a
generic fact-disruption channel.

Readout: joint log-probability of the frozen gold sequence (teacher forcing)
plus the top-1 token as a diagnostic. Compact scalars only; no raw
activations are persisted.

Usage (from the repo root):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/entity_cell_factual_amnesia_probe.py \
        --output /path/to/factual_amnesia_probe.json \
        --entity 'Johnson & Johnson' --cell 4,7676 --wrong-cell 0,1476 \
        --cross 'JPMorgan Chase' 'The Coca-Cola Company' 'Walmart Inc.' \
        --source-run artifacts/qwen3.5-4b/entity-cell-localization/runs/entity-cell-e1-hfm2-discovery-v1 \
        [--model .cache/models/qwen3.5-4b]
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from typing import Any

import torch

from llm_bias.core.continuation_scoring import _extract_logits, score_token_ids
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.encoding import decode_token, input_ids
from llm_bias.entity_cell.mlp_cells import (
    MLPHookSession,
    OnlineVectorStats,
    select_matched_random_neuron,
)

RUN = "artifacts/qwen3.5-4b/entity-cell-localization/runs/entity-cell-e1-hfm-discovery-v1"
DOSES = (1.0, 0.5, 0.0, -1.0, -2.0, -3.0)

# (label, frame, expected fact for human-side verification of the clean greedy)
FRAMES = (
    ("F0", "The headquarters of {name} is located in"),
    ("F2", "The stock ticker of {name} is"),
    ("F3", "{name} was founded in"),
)
END_DOSE = -3.0
GOLD_TOKENS = 3
GREEDY_TOKENS = 3


def next_log_distribution(model: Any, ids: list[int], device: Any) -> torch.Tensor:
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.forward(tensor)
        logits = _extract_logits(output, model, tensor).float()
    return torch.log_softmax(logits[0, -1], dim=-1)


def greedy_tokens(model: Any, ids: list[int], device: Any, tokenizer: Any, steps: int) -> list[int]:
    current = list(ids)
    out: list[int] = []
    for _ in range(steps):
        dist = next_log_distribution(model, current, device)
        token = int(dist.argmax())
        out.append(token)
        current.append(token)
    return out


def name_positions(tokenizer: Any, prompt: str, name: str) -> tuple[int, ...]:
    """Token indices whose span overlaps the entity-name char range."""
    start = prompt.index(name)
    end = start + len(name)
    encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"] if hasattr(encoded, "__getitem__") else encoded.offset_mapping
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    return tuple(i for i, (s, e) in enumerate(offsets) if e > start and s < end and (s, e) != (0, 0))


def hook_context(model: Any, cell: tuple[int, int] | None, dose: float, scope_positions: Any):
    if cell is None:
        return nullcontext()
    return MLPHookSession(
        model, [cell[0]],
        channel_scales={cell[0]: {cell[1]: float(dose)}},
        scope="all_positions" if scope_positions is None else "header_only",
        scope_positions=None if scope_positions is None else scope_positions,
    )


def run_probe(
    model: Any,
    tokenizer: Any,
    device: Any,
    stats: dict[int, OnlineVectorStats],
    *,
    entity: str,
    target_cell: tuple[int, int],
    wrong_cell: tuple[int, int],
    cross: list[str],
    source_run: str,
    label: str = "",
) -> dict[str, Any]:
    """Run the factual amnesia probe for one (entity, cell) target on a loaded model."""
    started = time.time()
    random_neuron = select_matched_random_neuron(stats, layer=target_cell[0], target_neuron=target_cell[1])

    rows: list[dict[str, Any]] = []
    gold_sequences: dict[str, dict[str, Any]] = {}

    def condition_row(prompt: str, group: str, label: str, gold_expect: str,
                      cell: tuple[int, int] | None, dose: float, scope_positions: Any,
                      gold_ids: list[int] | None, greedy: bool = False) -> dict[str, Any]:
        ids = input_ids(tokenizer, prompt, add_special_tokens=True)
        with hook_context(model, cell, dose, scope_positions):
            dist = next_log_distribution(model, ids, device)
            gold_joint = (
                float(score_token_ids(model, tokenizer, prompt, gold_ids, device=device).log_probability)
                if gold_ids else None
            )
            greedy_out = greedy_tokens(model, ids, device, tokenizer, GREEDY_TOKENS) if greedy else None
        top1 = int(dist.argmax())
        row: dict[str, Any] = {
            "group": group, "label": label, "prompt": prompt,
            "gold_expected": gold_expect,
            "cell": list(cell) if cell else None, "dose": dose if cell else None,
            "scope": "all_positions" if scope_positions is None else "name_positions",
            "top1_token": decode_token(tokenizer, top1),
            "top1_logp": float(dist[top1]),
        }
        if gold_ids is not None:
            row["gold_logp"] = gold_joint
        if greedy_out is not None:
            row["greedy_tokens"] = greedy_out
            row["greedy_text"] = tokenizer.decode(greedy_out, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        rows.append(row)
        return row

    # 1) Entity frames: freeze clean greedy as gold sequence, then sweep conditions.
    for frame, template in FRAMES:
        prompt = template.format(name=entity)
        ids = input_ids(tokenizer, prompt, add_special_tokens=True)
        gold_ids = greedy_tokens(model, ids, device, tokenizer, GOLD_TOKENS)
        gold_text = tokenizer.decode(gold_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        group = f"ENTITY_{frame}"
        gold_sequences[group] = {"tokens": gold_ids, "text": gold_text}
        print(f"{entity} {frame} clean gold={gold_text!r} (ids={gold_ids})", flush=True)
        condition_row(prompt, group, "clean", None, None, 1.0, None, gold_ids, greedy=True)
        for dose in DOSES:
            condition_row(prompt, group, "target", None, target_cell, dose, None, gold_ids,
                          greedy=(dose == END_DOSE))
        condition_row(prompt, group, "target_name_only", None, target_cell, END_DOSE,
                      name_positions(tokenizer, prompt, entity), gold_ids)
        condition_row(prompt, group, "wrong_entity", None, wrong_cell, END_DOSE, None, gold_ids)
        condition_row(prompt, group, "matched_random", None, (target_cell[0], random_neuron), END_DOSE, None, gold_ids)

    # 2) Cross-entity: same neuron on other entities' HQ facts (clean + end dose).
    for i, name in enumerate(cross):
        template = FRAMES[0][1]
        prompt = template.format(name=name)
        ids = input_ids(tokenizer, prompt, add_special_tokens=True)
        gold_ids = greedy_tokens(model, ids, device, tokenizer, GOLD_TOKENS)
        gold_text = tokenizer.decode(gold_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        group = f"CROSS{i}_{name.replace(' ', '_').replace('.', '')}"
        gold_sequences[group] = {"tokens": gold_ids, "text": gold_text}
        print(f"cross {name} F0 clean gold={gold_text!r} (ids={gold_ids})", flush=True)
        condition_row(prompt, group, "clean", None, None, 1.0, None, gold_ids, greedy=True)
        condition_row(prompt, group, "target", None, target_cell, END_DOSE, None, gold_ids)

    # 3) Compact per-frame collapse summary (end dose vs clean, gold-sequence joint logp).
    summary: dict[str, dict[str, float]] = {}
    for key in gold_sequences:
        clean_rows = [r for r in rows if r["label"] == "clean" and r["group"] == key]
        end_rows = [r for r in rows if r["label"] == "target" and r["group"] == key and r["dose"] == END_DOSE]
        if clean_rows and end_rows:
            clean_logp = clean_rows[0]["gold_logp"]
            end_logp = end_rows[0]["gold_logp"]
            summary[key] = {
                "clean_gold_logp": clean_logp,
                "end_gold_logp": end_logp,
                "collapse_nats": end_logp - clean_logp,
            }

    output = {
        "schema_version": 3,
        "artifact_type": "entity_cell_factual_amnesia_probe",
        "status": "proposed_probe_not_frozen",
        "label": label,
        "source_run": source_run,
        "device": str(device),
        "dtype": str(model._lm_head.weight.dtype),
        "target_cell": list(target_cell),
        "wrong_cell": list(wrong_cell),
        "matched_random_neuron": random_neuron,
        "entity": entity,
        "cross_entities": list(cross),
        "doses": list(DOSES),
        "gold_tokens": GOLD_TOKENS,
        "gold_sequences": gold_sequences,
        "frame_summary": summary,
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--entity", required=True, help="entity name surface form filling the frames")
    parser.add_argument("--cell", required=True, help="target cell as L,N (e.g. 0,1476)")
    parser.add_argument("--wrong-cell", required=True, help="wrong-entity control cell as L,N")
    parser.add_argument("--cross", nargs="+", required=True,
                        help="other entity names for the HQ cross-ticker control frames")
    parser.add_argument("--source-run", default=RUN,
                        help="discovery run the cell was localized in (provenance)")
    parser.add_argument("--baseline-stats", default=None,
                        help="baseline stats JSON; defaults to <source-run>/e1/baseline_stats.json")
    parser.add_argument("--label", default="", help="short label for provenance")
    args = parser.parse_args()
    target_cell = tuple(int(part) for part in args.cell.split(","))
    wrong_cell = tuple(int(part) for part in args.wrong_cell.split(","))
    baseline_stats = args.baseline_stats or f"{args.source_run}/e1/baseline_stats.json"

    model, tokenizer, device = load_model(args.model)
    with open(baseline_stats) as fh:
        stats_payload = json.load(fh)
    stats = {int(layer): OnlineVectorStats.from_compact(value) for layer, value in stats_payload["layers"].items()}

    output = run_probe(
        model, tokenizer, device, stats,
        entity=args.entity, target_cell=target_cell, wrong_cell=wrong_cell,
        cross=args.cross, source_run=args.source_run, label=args.label,
    )
    output["model"] = args.model
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=1)
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
