"""CLI for the counterfactual residual activation patching experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_bias.core.model import DEFAULT_MODEL
from llm_bias.counterfactual_patching.experiment import (
    prepare_data,
    run_patch,
    summarize,
    visualize,
)

DEFAULT_ROOT = "artifacts/counterfactual_patching"
BINARY_DATASET = "easy-bias-zh-tw-binary-v1"


def _model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)


def _manifest_for(args: argparse.Namespace):
    manifest_path = getattr(args, "manifest", None)
    if not manifest_path:
        return None
    from llm_bias.core.artifact_manifest import RunManifest

    path = Path(manifest_path)
    if path.exists():
        manifest = RunManifest.load(path)
        if manifest.status == "created":
            manifest.start()
    else:
        manifest = RunManifest(
            model=args.model,
            dataset=BINARY_DATASET,
            run_id=path.parent.name,
            run_directory=path.parent,
        )
        manifest.start()
    manifest.save()
    return manifest


def _finish_manifest_stage(
    manifest,
    *,
    stage: str,
    inputs: list[tuple[Path, str] | tuple[Path, str, str]],
    outputs: list[tuple[Path, str, str, int | None]],
) -> None:
    if manifest is None:
        return
    manifest.start_stage(stage)
    for item in inputs:
        path, artifact_type = item[:2]
        role = item[2] if len(item) == 3 else "input"
        manifest.register_artifact(
            path,
            artifact_type=artifact_type,
            stage=stage,
            role=role,
        )
    for path, artifact_type, role, record_count in outputs:
        manifest.register_artifact(
            path,
            artifact_type=artifact_type,
            stage=stage,
            role=role,
            record_count=record_count,
        )
    manifest.finish_stage(stage)
    manifest.save()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-data")
    _model_argument(prepare)
    prepare.add_argument("--output", default=f"{DEFAULT_ROOT}/pairs.jsonl")
    prepare.add_argument("--max-pairs", type=int)

    run = commands.add_parser("run")
    _model_argument(run)
    run.add_argument("--pairs", default=f"{DEFAULT_ROOT}/pairs.jsonl")
    run.add_argument("--lens", required=True)
    run.add_argument("--output", default=f"{DEFAULT_ROOT}/patch_results.jsonl")
    run.add_argument("--max-pairs", type=int)

    binary_prepare = commands.add_parser("prepare-binary-association")
    _model_argument(binary_prepare)
    binary_prepare.add_argument("--output-dir", default="artifacts/easy-bias-zh-tw-binary-v1/prepare")
    binary_prepare.add_argument("--careers")
    binary_prepare.add_argument("--inference")
    binary_prepare.add_argument("--order")
    binary_prepare.add_argument("--seed", type=int, default=0)
    binary_prepare.add_argument("--max-careers", type=int)
    binary_prepare.add_argument("--manifest")

    binary_baseline = commands.add_parser("baseline-binary-association")
    _model_argument(binary_baseline)
    binary_baseline.add_argument("--rendered", required=True)
    binary_baseline.add_argument("--output", default="artifacts/easy-bias-zh-tw-binary-v1/baseline_scores.jsonl")
    binary_baseline.add_argument("--max-rows", type=int)
    binary_baseline.add_argument("--manifest")

    binary_patch = commands.add_parser("run-binary-patch")
    _model_argument(binary_patch)
    binary_patch.add_argument("--pairs", required=True)
    binary_patch.add_argument("--layer", type=int, required=True)
    binary_patch.add_argument("--output", default="artifacts/easy-bias-zh-tw-binary-v1/patch_results.jsonl")
    binary_patch.add_argument("--max-pairs", type=int)
    binary_patch.add_argument("--manifest")

    fit_direction = commands.add_parser("fit-binary-direction")
    _model_argument(fit_direction)
    fit_direction.add_argument("--baseline", required=True)
    fit_direction.add_argument("--layer", type=int, required=True)
    fit_direction.add_argument("--output", default="artifacts/easy-bias-zh-tw-binary-v1/direction.pt")
    fit_direction.add_argument("--manifest")

    run_steering = commands.add_parser("run-binary-steering")
    _model_argument(run_steering)
    run_steering.add_argument("--baseline", required=True)
    run_steering.add_argument("--direction", required=True)
    run_steering.add_argument("--output", default="artifacts/easy-bias-zh-tw-binary-v1/steering_results.jsonl")
    run_steering.add_argument("--alpha", type=float, nargs="+", required=True)
    run_steering.add_argument("--split", default="confirmation")
    run_steering.add_argument(
        "--direction-variant",
        choices=("fitted", "random", "permuted"),
        default="fitted",
    )
    run_steering.add_argument(
        "--span-policy",
        choices=("entity", "final-non-entity"),
        default="entity",
    )
    run_steering.add_argument("--seed", type=int, default=0)
    run_steering.add_argument("--manifest")

    binary_summary = commands.add_parser("summarize-binary-association")
    binary_summary.add_argument("--baseline", required=True)
    binary_summary.add_argument("--patch")
    binary_summary.add_argument("--steering")
    binary_summary.add_argument(
        "--output",
        default="artifacts/easy-bias-zh-tw-binary-v1/binary_summary.json",
    )
    binary_summary.add_argument("--seed", type=int, default=0)
    binary_summary.add_argument("--resamples", type=int, default=10_000)
    binary_summary.add_argument("--manifest")

    lens_check = commands.add_parser("validate-binary-lens")
    _model_argument(lens_check)
    lens_check.add_argument("--lens", required=True)
    lens_check.add_argument("--output", required=True)
    lens_check.add_argument("--manifest")

    finalize_manifest = commands.add_parser("finalize-binary-manifest")
    finalize_manifest.add_argument("--manifest", required=True)

    summary = commands.add_parser("summarize")
    summary.add_argument("--input", default=f"{DEFAULT_ROOT}/patch_results.jsonl")
    summary.add_argument("--output-dir", default=DEFAULT_ROOT)

    visual = commands.add_parser("visualize")
    visual.add_argument("--input", default=f"{DEFAULT_ROOT}/patch_results.jsonl")
    visual.add_argument("--output-dir", default=DEFAULT_ROOT)

    serve = commands.add_parser("serve")
    _model_argument(serve)
    serve.add_argument(
        "--lens",
        help="defaults to artifacts/<model-slug>/jacobian-lens/jacobian_lens.pt",
    )
    serve.add_argument("--pairs", default=f"{DEFAULT_ROOT}/pairs.jsonl")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8321)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        prepare_data(args.model, args.output, args.max_pairs)
    elif args.command == "run":
        run_patch(args.model, args.pairs, args.lens, args.output, args.max_pairs)
    elif args.command == "prepare-binary-association":
        from llm_bias.counterfactual_patching.binary_association import prepare_binary_association
        from llm_bias.core.model import load_tokenizer

        tokenizer = load_tokenizer(args.model)
        manifest = _manifest_for(args)
        prepare_binary_association(
            tokenizer,
            output_dir=args.output_dir,
            careers_path=args.careers,
            inference_path=args.inference,
            order_path=args.order,
            seed=args.seed,
            max_careers=args.max_careers,
        )
        output_dir = Path(args.output_dir)
        _finish_manifest_stage(
            manifest,
            stage="prepare",
            inputs=[],
            outputs=[
                (output_dir / "careers.jsonl", "careers", "output", None),
                (output_dir / "rendered_prompts.jsonl", "rendered_prompts", "output", None),
                (output_dir / "pairs.jsonl", "pairs", "output", None),
                (output_dir / "omissions.jsonl", "omissions", "output", None),
                (output_dir / "prepare_metadata.json", "prepare_metadata", "output", None),
            ],
        )
    elif args.command == "baseline-binary-association":
        from llm_bias.counterfactual_patching.binary_runner import run_baseline
        from llm_bias.core.model import load_model

        model, tokenizer, device = load_model(args.model)
        manifest = _manifest_for(args)
        run_baseline(
            model,
            tokenizer,
            args.rendered,
            args.output,
            device=device,
            max_rows=args.max_rows,
        )
        _finish_manifest_stage(
            manifest,
            stage="baseline",
            inputs=[(Path(args.rendered), "rendered_prompts")],
            outputs=[(Path(args.output), "baseline_scores", "output", None)],
        )
    elif args.command == "run-binary-patch":
        import json
        from itertools import islice

        from llm_bias.counterfactual_patching.binary_association import iter_binary_pairs
        from llm_bias.counterfactual_patching.binary_runner import patch_pair_single_layer
        from llm_bias.core.model import load_model

        model, tokenizer, device = load_model(args.model)
        manifest = _manifest_for(args)
        pairs = iter_binary_pairs(args.pairs)
        if args.max_pairs is not None:
            pairs = islice(pairs, args.max_pairs)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for pair in pairs:
                row = patch_pair_single_layer(
                    model, tokenizer, pair, layer=args.layer, device=device
                )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        _finish_manifest_stage(
            manifest,
            stage="patch",
            inputs=[(Path(args.pairs), "pairs")],
            outputs=[(output, "patch_results", "output", None)],
        )
    elif args.command == "fit-binary-direction":
        from llm_bias.counterfactual_patching.artifact_io import read_jsonl
        from llm_bias.counterfactual_patching.steering import fit_direction, save_direction
        from llm_bias.core.model import load_model

        model, tokenizer, device = load_model(args.model)
        manifest = _manifest_for(args)
        rows = [row for row in read_jsonl(args.baseline) if row.get("split") == "train"]
        direction, metadata = fit_direction(
            model,
            rows,
            layer=args.layer,
            device=device,
            model_name=args.model,
            tokenizer_name=getattr(tokenizer, "name_or_path", None),
        )
        save_direction(args.output, direction, metadata)
        _finish_manifest_stage(
            manifest,
            stage="direction",
            inputs=[(Path(args.baseline), "baseline_scores")],
            outputs=[
                (Path(args.output), "direction_vector", "output", None),
                (Path(args.output).with_suffix(".metadata.json"), "direction_metadata", "output", None),
            ],
        )
    elif args.command == "run-binary-steering":
        import json
        from llm_bias.counterfactual_patching.artifact_io import read_jsonl
        from llm_bias.counterfactual_patching.binary_runner import (
            _last_non_entity_position,
            _span_tuples,
            steer_pair,
        )
        from llm_bias.counterfactual_patching.steering import (
            load_direction,
            norm_matched_random_direction,
            permuted_direction,
        )
        from llm_bias.core.model import load_model

        model, tokenizer, device = load_model(args.model)
        manifest = _manifest_for(args)
        direction, metadata = load_direction(args.direction)
        if metadata.model is not None and metadata.model != args.model:
            raise ValueError("direction model does not match requested model")
        tokenizer_name = getattr(tokenizer, "name_or_path", None)
        if metadata.tokenizer is not None and metadata.tokenizer != tokenizer_name:
            raise ValueError("direction tokenizer does not match requested tokenizer")
        if args.direction_variant == "random":
            direction = norm_matched_random_direction(direction, seed=args.seed)
        elif args.direction_variant == "permuted":
            direction = permuted_direction(direction)
        rows = [row for row in read_jsonl(args.baseline) if row.get("split") == args.split]
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                source_spans = None
                if args.span_policy == "final-non-entity":
                    spans = _span_tuples(row)
                    position = _last_non_entity_position(spans, len(row["input_ids"]))
                    source_spans = [(position, position + 1)]
                for alpha in args.alpha:
                    result = steer_pair(
                        model,
                        tokenizer,
                        row,
                        layer=metadata.layer,
                        direction=direction,
                        alpha=alpha,
                        device=device,
                        source_spans=source_spans,
                    )
                    result["direction_variant"] = args.direction_variant
                    result["span_policy"] = args.span_policy
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        _finish_manifest_stage(
            manifest,
            stage="steering",
            inputs=[
                (Path(args.baseline), "baseline_scores"),
                (Path(args.direction), "direction_vector"),
                (Path(args.direction).with_suffix(".metadata.json"), "direction_metadata"),
            ],
            outputs=[(output, "steering_results", "output", None)],
        )
    elif args.command == "summarize-binary-association":
        import json
        from llm_bias.counterfactual_patching.artifact_io import read_jsonl
        from llm_bias.counterfactual_patching.binary_summary import summarize_binary

        manifest = _manifest_for(args)
        result = summarize_binary(
            read_jsonl(args.baseline),
            patch_rows=read_jsonl(args.patch) if args.patch else None,
            steering_rows=read_jsonl(args.steering) if args.steering else None,
            seed=args.seed,
            n_resamples=args.resamples,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        inputs = [(Path(args.baseline), "baseline_scores")]
        if args.patch:
            inputs.append((Path(args.patch), "patch_results"))
        if args.steering:
            inputs.append((Path(args.steering), "steering_results"))
        _finish_manifest_stage(
            manifest,
            stage="summary",
            inputs=inputs,
            outputs=[(output, "binary_summary", "output", None)],
        )
    elif args.command == "validate-binary-lens":
        import json
        from llm_bias.counterfactual_patching.binary_runner import validate_binary_lens
        from llm_bias.core.model import load_model

        model, _tokenizer, _device = load_model(args.model)
        manifest = _manifest_for(args)
        result = validate_binary_lens(
            model,
            model_name=args.model,
            lens_path=args.lens,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        _finish_manifest_stage(
            manifest,
            stage="lens-validation",
            inputs=[(Path(args.lens), "jacobian_lens", "lens")],
            outputs=[(output, "lens_validation", "output", None)],
        )
    elif args.command == "finalize-binary-manifest":
        from llm_bias.core.artifact_manifest import RunManifest

        manifest = RunManifest.load(args.manifest)
        manifest.complete()
        manifest.save()
    elif args.command == "summarize":
        summarize(args.input, args.output_dir)
    elif args.command == "visualize":
        visualize(args.input, args.output_dir)
    elif args.command == "serve":
        import uvicorn

        from llm_bias.counterfactual_patching.visualization import build_app

        app = build_app(args.model, args.lens, args.pairs)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
