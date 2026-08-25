"""CLI for running baseline trial-plan prompts through prompt-analysis stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_bias.baseline_trial.pipeline import (
    DEFAULT_INPUT,
    DEFAULT_MODEL,
    STAGES,
    run_baseline_trial,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser(
        "inspect-input",
        help="validate baseline trial-plan CSV compatibility without model inference",
    )
    inspect.add_argument("--input", default=DEFAULT_INPUT)
    inspect.add_argument("--output", help="write the machine-readable JSON report here")

    run = subparsers.add_parser(
        "run",
        help="execute the baseline trial-plan stages into one canonical run tree",
    )
    run.add_argument("--input", default=DEFAULT_INPUT)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument(
        "--lens",
        default=None,
        help="defaults to artifacts/<model-slug>/jacobian-lens/jacobian_lens.pt",
    )
    run.add_argument(
        "--dataset",
        default=None,
        help="dataset identity; defaults to the input filename stem",
    )
    run.add_argument("--run-id", required=True)
    run.add_argument("--artifact-root", default="artifacts")
    run.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        dest="stages",
        help="enabled stage; repeatable. Default: readout forward lens-forward backward validate",
    )
    run.add_argument("--max-seq-len", type=int, default=1024)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--top-k", type=int, default=15)
    run.add_argument("--max-rows", type=int, default=None)
    run.add_argument("--prompt-column", action="append", dest="prompt_columns")
    run.add_argument("--generate-full", action="store_true", default=True)
    run.add_argument(
        "--sample-per-condition",
        type=int,
        default=32,
        help="ignored when --generate-full (default); 0 is rejected, use --generate-full",
    )
    run.add_argument("--max-new-tokens", type=int, default=256)
    run.add_argument(
        "--device-map",
        choices=("qwen27b_two_gpu", "qwen27b_two_gpu_24", "qwen27b_two_gpu_16"),
        default=None,
        help=(
            "shard the model across two GPUs (baseline 27B on busy machines); "
            "_24/_16 keep fewer layers on GPU 0 so the Jacobian lens fits "
            "on the lm_head GPU (GPU 1)"
        ),
    )
    run.add_argument("--backward-input-top-k", type=int, default=None)
    run.add_argument("--backward-output-token-top-k", type=int, default=None)
    run.add_argument(
        "--lens-forward-layers",
        default=None,
        help="comma-separated lens layer subset for the lens-forward stage (default: all source layers)",
    )
    run.add_argument(
        "--forward-artifact",
        default=None,
        help="existing forward JSONL for lens-forward/backward/validate without re-running forward",
    )
    run.add_argument("--validate-seed", type=int, default=0)
    return parser


def main() -> None:
    import json

    args = build_parser().parse_args()
    if args.command == "inspect-input":
        from llm_bias.prompt_analysis.input_inspection import inspect_input_to_json

        report = inspect_input_to_json(args.input, args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["errors"]:
            raise SystemExit(1)
        return

    stages = tuple(args.stages) if args.stages else tuple(STAGES)
    lens_path: Path | None
    if args.lens is not None:
        lens_path = Path(args.lens)
    else:
        from llm_bias.core.artifact_paths import jacobian_lens_path

        lens_path = jacobian_lens_path(args.model, artifact_root=args.artifact_root)
        if not lens_path.is_file():
            lens_path = None
    run_root = run_baseline_trial(
        input_path=args.input,
        model=args.model,
        lens=lens_path,
        dataset=args.dataset,
        run_id=args.run_id,
        artifact_root=args.artifact_root,
        stages=stages,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        top_k=args.top_k,
        max_rows=args.max_rows,
        prompt_columns=args.prompt_columns,
        generate_full=args.generate_full,
        sample_per_condition=args.sample_per_condition,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        backward_input_top_k=args.backward_input_top_k,
        backward_output_token_top_k=args.backward_output_token_top_k,
        lens_forward_layers=(
            [int(layer) for layer in args.lens_forward_layers.split(",") if layer.strip()]
            if args.lens_forward_layers
            else None
        ),
        forward_artifact=args.forward_artifact,
        validate_seed=args.validate_seed,
    )
    print(run_root)


if __name__ == "__main__":
    main()
