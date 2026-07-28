"""CLI for fitting a reusable Jacobian-lens artifact."""

from __future__ import annotations

import argparse

from llm_bias.core.model import DEFAULT_MODEL
from llm_bias.lens_fitting.fitting import fit_jacobian_lens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="artifacts/lenses/jacobian_lens.pt")
    parser.add_argument("--calibration-prompts", type=int, default=16)
    parser.add_argument("--calibration-file")
    parser.add_argument("--calibration-field", default="text")
    parser.add_argument("--layer-stride", type=int, default=2)
    parser.add_argument("--dim-batch", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fit_jacobian_lens(
        model_name=args.model,
        output=args.output,
        calibration_count=args.calibration_prompts,
        calibration_file=args.calibration_file,
        calibration_field=args.calibration_field,
        layer_stride=args.layer_stride,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
    )


if __name__ == "__main__":
    main()
