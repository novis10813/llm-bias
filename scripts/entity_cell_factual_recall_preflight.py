"""Entity-cell factual recall preflight (proposed; not a frozen protocol).

Go/no-go check for the proposed factual amnesia probe: does the model hold
entity-specific parametric factual completions for the E1 V2 trusted-candidate
ticker (FTNT) and the E3 control tickers (ADI, MU, FTV)?

Prompt-driven by design: it reuses the frozen E1 V2 factual-cloze frame
wordings (llm_bias.entity_cell.preparation.FRAME_VARIANT_SPECS, variant ids
F0/F2/F3/H0) and scores the model's own clean-baseline continuations. No
external fact values are injected into the model; the compact top-k and
greedy-continuation outputs are reviewed human-side against web-grounded
facts to set the go/no-go verdict.

Entity surfaces follow data/all_constituents_2020_2025.csv as used by the
entity-cell experiments: FTNT=Fortinet, ADI=Analog Devices,
MU=Micron Technology, FTV=Fortive. An ad-hoc screen can override the entity
set via --entities (one TICKER:Name shell argument per ticker; multiple
surface variants per ticker separated by '|'). The anonymous control is
scored once per frame, not per entity.

Device: load_model uses cuda:0 (bfloat16) when CUDA is available, else CPU
(fp32). Select the physical card with CUDA_VISIBLE_DEVICES, e.g.
`CUDA_VISIBLE_DEVICES=0 uv run python scripts/entity_cell_factual_recall_preflight.py ...`.

Definition and status: see docs/entity-cell-localization/details/README.md
("Proposed: factual recall preflight").

Usage (from the repo root):
    uv run python scripts/entity_cell_factual_recall_preflight.py \
        --output /path/to/factual_recall_preflight.json \
        [--model .cache/models/qwen3.5-4b]
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
import transformers

from llm_bias.core.continuation_scoring import _extract_logits
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.encoding import decode_token, format_prompt, input_ids

# Frozen E1 V2 factual-cloze frames: (variant_id, template).
CLOZE_FRAMES = (
    (0, "The headquarters of {name} is located in"),
    (2, "The stock ticker of {name} is"),
    (3, "{name} was founded in"),
    (8, "The CEO of {name} is"),
)
# Entity surfaces as used by the entity-cell experiments.
ENTITIES = {
    "FTNT": "Fortinet",
    "ADI": "Analog Devices",
    "MU": "Micron Technology",
    "FTV": "Fortive",
}
ANONYMOUS_NAME = "Anonymous Company"
TOP_K = 5
GREEDY_TOKENS = 5


def next_log_distribution(model: Any, ids: list[int], device: Any) -> torch.Tensor:
    """Full-vocabulary next-token log distribution for one id sequence."""
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.forward(tensor)
        logits = _extract_logits(output, model, tensor).float()
    return torch.log_softmax(logits[0, -1], dim=-1)


def model_dtype(model: Any) -> str:
    """Read the weight dtype without relying on nn.Module.parameters()."""
    for name in ("_lm_head", "lm_head"):
        head = getattr(model, name, None)
        if head is not None and hasattr(head, "weight"):
            return str(head.weight.dtype)
    layers = getattr(model, "layers", None)
    if layers is not None:
        first = layers[0]
        for value in vars(first).values():
            if hasattr(value, "weight") and value.weight is not None:
                return str(value.weight.dtype)
    return "unknown"


def parse_entities(spec: list[str] | None) -> dict[str, list[str]]:
    if not spec:
        return {t: [n] for t, n in ENTITIES.items()}
    out: dict[str, list[str]] = {}
    for token in spec:
        ticker, names = token.split(":", 1)
        out[ticker] = [n for n in names.split("|") if n]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tickers", nargs="+", default=None, choices=None)
    parser.add_argument("--entities", nargs="+", default=None,
                        help="TICKER:Name arguments ('|' joins surface variants); overrides the built-in set")
    args = parser.parse_args()

    entities = parse_entities(args.entities)
    if args.tickers:
        unknown = [t for t in args.tickers if t not in entities]
        if unknown:
            raise SystemExit(f"unknown tickers for --tickers: {unknown}")
        entities = {t: entities[t] for t in args.tickers}

    started = time.time()
    model, tokenizer, device = load_model(args.model)
    rows: list[dict] = []

    def score(surface: str) -> list[dict]:
        frame_rows = []
        for variant_id, template in CLOZE_FRAMES:
            prompt = template.format(name=surface)
            ids = input_ids(tokenizer, prompt, add_special_tokens=True)
            dist = next_log_distribution(model, ids, device)
            top_k = [
                {"token_id": int(i), "text": decode_token(tokenizer, int(i)), "logp": float(dist[i])}
                for i in dist.topk(TOP_K).indices.tolist()
            ]
            greedy_ids: list[int] = []
            current = list(ids)
            for _ in range(GREEDY_TOKENS):
                step = next_log_distribution(model, current, device)
                token = int(step.argmax())
                greedy_ids.append(token)
                current.append(token)
            frame_rows.append({
                "surface": surface,
                "frame_variant": variant_id,
                "frame_template": template,
                "prompt": prompt,
                "top_k": top_k,
                "greedy_tokens": greedy_ids,
                "greedy_text": tokenizer.decode(greedy_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            })
            print(
                f"{surface:28s} F{variant_id} | "
                f"top1={top_k[0]['text']!r} ({top_k[0]['logp']:+.3f}) | "
                f"greedy={frame_rows[-1]['greedy_text']!r}",
                flush=True,
            )
        return frame_rows

    # Target surfaces (all variants of all tickers).
    for ticker, surfaces in entities.items():
        for surface in surfaces:
            for frame_row in score(surface):
                rows.append({"ticker": ticker, "label": "target", **frame_row})
    # Anonymous control, once per frame.
    for frame_row in score(ANONYMOUS_NAME):
        rows.append({"ticker": None, "label": "anonymous", **frame_row})

    output = {
        "schema_version": 2,
        "artifact_type": "entity_cell_factual_recall_preflight",
        "status": "proposed_preflight_not_frozen",
        "model": args.model,
        "device": str(device),
        "dtype": model_dtype(model),
        "transformers_version": transformers.__version__,
        "tickers": sorted(entities),
        "entities": entities,
        "frames": [{"variant_id": v, "template": t} for v, t in CLOZE_FRAMES],
        "top_k": TOP_K,
        "greedy_tokens": GREEDY_TOKENS,
        "elapsed_seconds": round(time.time() - started, 1),
        "rows": rows,
    }
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=1)
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
