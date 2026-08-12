"""CLI for installing exact, pinned pretrained Jacobian lenses."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from llm_bias.core.lens_registry import DEFAULT_PRETRAINED_LENS_REGISTRY
from llm_bias.lens_install.importer import install_pretrained_lens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install a validated pinned pretrained Jacobian lens"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--base-model",
        help="Canonical Hub identity for a local checkpoint whose config does not record one",
    )
    parser.add_argument(
        "--registry", default=str(DEFAULT_PRETRAINED_LENS_REGISTRY)
    )
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument(
        "--artifact-model-name",
        help="Model identity used only for the canonical artifact directory",
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
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
