"""Evidence-insensitivity Phase 1 operator."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from llm_bias.evidence_insensitivity import pipeline


def _default_run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--run-id", default=None, help="default: pilot-<UTC timestamp>")
    pilot.add_argument("--model", default=".cache/models/qwen3.5-4b")
    pilot.add_argument("--model-slug", default=None, help="artifact root slug (default: qwen3.5-4b)")
    pilot.add_argument("--artifact-root", default="artifacts")
    for name in ("prepare", "forward", "analyze"):
        command = sub.add_parser(name)
        command.add_argument("--run-id", required=True)
        command.add_argument("--model", default=".cache/models/qwen3.5-4b")
        command.add_argument("--model-slug", default=None, help="artifact root slug (default: qwen3.5-4b)")
        command.add_argument("--artifact-root", default="artifacts")
    return root


def main() -> None:
    args = parser().parse_args()
    root = Path(args.artifact_root)
    run_id = args.run_id
    if args.command == "pilot":
        run_id = run_id or _default_run_id("pilot")
        pipeline.run_pilot(run_id, artifact_root=root, model_path=args.model, model_slug=args.model_slug)
    elif args.command == "prepare":
        pipeline.run_prepare(run_id, artifact_root=root, model_path=args.model, model_slug=args.model_slug)
    elif args.command == "forward":
        pipeline.run_forward(run_id, artifact_root=root, model_path=args.model, model_slug=args.model_slug)
    else:
        pipeline.run_analyze(run_id, artifact_root=root, model_slug=args.model_slug)
    print(root / (args.model_slug or "qwen3.5-4b") / "evidence-insensitivity" / "runs" / run_id)


if __name__ == "__main__":
    main()
