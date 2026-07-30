#!/usr/bin/env python
"""Evaluate three Qwen Jacobian-lens candidates on the bilingual holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_bias.lens_fitting.evaluation import evaluate_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("artifacts/candidate_lenses/qwen3.5-4b"),
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=Path(
            "data/evaluations/qwen3.5-4b/"
            "bilingual_intermediate_holdout.jsonl"
        ),
    )
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--expected-calibration-prompts", type=int, default=128)
    parser.add_argument("--band-start", type=int)
    parser.add_argument("--band-end", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidates = {
        condition: args.candidate_root / condition / "jacobian_lens.pt"
        for condition in ("english", "chinese_simplified", "mixed")
    }
    missing = [str(path) for path in candidates.values() if not path.is_file()]
    if missing:
        parser.error(f"candidate lenses are missing: {', '.join(missing)}")
    output = args.output or args.candidate_root / "evaluation.json"
    result = evaluate_candidates(
        model_name=args.model,
        candidate_paths=candidates,
        holdout_path=args.holdout,
        output_path=output,
        max_seq_len=args.max_seq_len,
        use_chat_template=True,
        band_start=args.band_start,
        band_end=args.band_end,
        expected_n_prompts=args.expected_calibration_prompts,
    )
    summaries = {
        name: value["summary"] for name, value in result["candidates"].items()
    }
    print(
        json.dumps(
            {
                "selected_candidate": result["selected_candidate"],
                "layer_band": result["layer_band"],
                "selection_uncertainty": result["selection_uncertainty"],
                "summaries": summaries,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
