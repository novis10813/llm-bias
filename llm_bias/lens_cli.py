"""Unified CLI for fitting and installing Jacobian-lens artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from llm_bias.core.lens_artifacts import canonical_lens_path
from llm_bias.core.lens_registry import DEFAULT_PRETRAINED_LENS_REGISTRY
from llm_bias.core.model import DEFAULT_MODEL
from llm_bias.lens_fitting.fitting import fit_jacobian_lens
from llm_bias.lens_install.importer import install_pretrained_lens


def _add_fit_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("fit", help="Fit a model-specific lens")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        help="defaults to artifacts/<model-slug>/jacobian-lens/jacobian_lens.pt",
    )
    parser.add_argument("--calibration-prompts", type=int, default=16)
    parser.add_argument("--calibration-file")
    parser.add_argument("--calibration-field", default="text")
    parser.add_argument("--layer-stride", type=int, default=1)
    parser.add_argument("--dim-batch", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--device-map", choices=("balanced", "qwen27b_two_gpu"), default=None
    )
    parser.add_argument("--device-map-json")
    parser.add_argument("--max-memory-json")
    parser.add_argument("--selection-basis")
    parser.add_argument("--checkpoint-every", type=int, default=4)


def _add_install_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "install", help="Install an exact pinned pretrained lens"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--registry", default=str(DEFAULT_PRETRAINED_LENS_REGISTRY))
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--artifact-model-name")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_fit_parser(subparsers)
    _add_install_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        fit_jacobian_lens(
            model_name=args.model,
            output=args.output or str(canonical_lens_path(args.model)),
            calibration_count=args.calibration_prompts,
            calibration_file=args.calibration_file,
            calibration_field=args.calibration_field,
            layer_stride=args.layer_stride,
            dim_batch=args.dim_batch,
            max_seq_len=args.max_seq_len,
            skip_first=args.skip_first,
            use_chat_template=args.chat_template,
            enable_thinking=args.enable_thinking,
            checkpoint_every=args.checkpoint_every,
            device_map=(
                json.loads(args.device_map_json)
                if args.device_map_json
                else args.device_map
            ),
            max_memory=(
                json.loads(args.max_memory_json) if args.max_memory_json else None
            ),
            selection_basis=args.selection_basis,
        )
        return
    result = install_pretrained_lens(
        model_name=args.model,
        base_model=args.base_model,
        registry_path=args.registry,
        artifact_root=args.artifact_root,
        artifact_model_name=args.artifact_model_name,
        cache_dir=args.cache_dir,
        offline=args.offline,
        dry_run=args.dry_run,
        replace_existing=args.replace_existing,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
