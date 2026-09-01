"""CLI for entity-cell preparation, E1/E2, and E3 discovery."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="validate and materialize tokenizer-only inputs")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--split-manifest", type=Path, required=True)
    prepare.add_argument("--baseline", type=Path, required=True)
    prepare.add_argument("--baseline-identity", required=True)
    prepare.add_argument("--model", required=True, help="model identity for provenance; no model is loaded")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    prepare.add_argument("--dataset", default="entity-cell-localization")
    prepare.add_argument("--split", choices=("discovery", "calibration", "test"), default="discovery")
    prepare.add_argument("--sector", default="Technology")
    prepare.add_argument("--baseline-count", type=int, default=399)
    prepare.add_argument("--tokenizer", default=None, help="tokenizer identity/path; defaults to --model")
    run = commands.add_parser("run", help="run E1 localization/amnesia, E2 attribution, or E3 discovery from prepared inputs")
    run.add_argument("--prepared-dir", type=Path, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    run.add_argument("--stage", action="append", choices=("e1-baseline", "e1-localization", "e1-amnesia", "e2-attribution", "e2-readout", "e2-patching", "e3-upstream", "e3-downstream", "analyze"), dest="stages")
    run.add_argument("--e2-layers", nargs="+", type=int, default=None, help="E2 full-attention layers; defaults to L3/L7/L11/L15/L19/L23/L27/L31")
    run.add_argument("--e3-e1-run-root", type=Path, default=None, help="completed E1 discovery run used for trusted cells")
    run.add_argument("--e3-e2-run-root", type=Path, default=None, help="completed E2 discovery run used for selected heads")
    run.add_argument("--e3-head", action="append", nargs=2, type=int, metavar=("LAYER", "HEAD"), default=None, help="selected E2 head; repeat for a group")
    run.add_argument("--e3-grouping", choices=("single", "group"), default="single")
    run.add_argument("--max-tickers", type=int, default=None, help="one-ticker smoke cap when set to 1")
    run.add_argument("--lens-path", type=Path, default=None, help="explicit validated canonical lens path")
    run.add_argument("--expected-lens-sha256", default=None)
    run.add_argument("--expected-model-revision", default=None)
    run.add_argument("--expected-tokenizer-identity", default=None)
    confirmation_config = commands.add_parser("prepare-confirmation-config", help="freeze the V1 confirmation config from discovery selections")
    confirmation_config.add_argument("--output", type=Path, required=True)
    confirmation_config.add_argument("--selected-head", action="append", nargs=2, type=int, metavar=("LAYER", "HEAD"), required=True)
    confirmation_config.add_argument("--minimum-eligible-tickers", type=int, default=8)
    confirmation_config.add_argument("--model", default=None)
    confirmation_config.add_argument("--discovery-run", type=Path, default=None)
    confirmation_config.add_argument("--split-manifest-sha256", default=None)
    confirmation = commands.add_parser("analyze-confirmation", help="evaluate frozen calibration/test confirmation records")
    confirmation.add_argument("--records", type=Path, required=True)
    confirmation.add_argument("--config", type=Path, required=True)
    confirmation.add_argument("--output", type=Path, required=True)
    confirmation.add_argument("--split", choices=("calibration", "test"), required=True)
    confirmation.add_argument("--calibration", type=Path, default=None)
    analyze = commands.add_parser("analyze", help="analyze completed compact E1, E2, E3, or discovery outputs")
    analyze.add_argument("--run-root", type=Path, required=False)
    analyze.add_argument("--experiment", choices=("e1", "e2", "e3", "discovery"), default="e1")
    analyze.add_argument("--e1-run-root", type=Path, default=None)
    analyze.add_argument("--e2-run-root", type=Path, default=None)
    analyze.add_argument("--e3-run-root", type=Path, default=None)
    analyze.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-confirmation-config":
        import json
        from llm_bias.entity_cell.confirmation import default_confirmation_config
        from llm_bias.core.artifact_paths import sha256_file
        discovery = None if args.discovery_run is None else str(args.discovery_run)
        discovery_hash = None if args.discovery_run is None else sha256_file(args.discovery_run / "manifest.json")
        payload = default_confirmation_config(selected_heads=args.selected_head, minimum_eligible_tickers=args.minimum_eligible_tickers, discovery_run=discovery, discovery_run_sha256=discovery_hash, split_manifest_sha256=args.split_manifest_sha256, model=args.model)
        from llm_bias.core.artifacts.io import write_json
        write_json(args.output, payload, overwrite=False)
        root = args.output
    elif args.command == "analyze-confirmation":
        from llm_bias.entity_cell.confirmation import evaluate_confirmation_artifact
        root = args.output
        evaluate_confirmation_artifact(args.records, args.config, args.output, split=args.split, calibration_path=args.calibration)
    elif args.command == "prepare":
        from llm_bias.core.model import load_tokenizer
        from llm_bias.entity_cell.preparation import prepare_artifacts
        root = prepare_artifacts(
            input_path=args.input, split_manifest=args.split_manifest, baseline_source=args.baseline,
            baseline_identity=args.baseline_identity, tokenizer=load_tokenizer(args.tokenizer or args.model),
            model=args.model, run_id=args.run_id, artifact_root=args.artifact_root, dataset=args.dataset,
            split=args.split, sector=args.sector, baseline_expected_count=args.baseline_count,
        )
    elif args.command == "run":
        stages = tuple(args.stages) if args.stages else ("e1-baseline", "e1-localization", "e1-amnesia", "analyze")
        if any(stage.startswith("e3-") for stage in stages):
            from llm_bias.entity_cell.e3 import run_e3
            root = run_e3(prepared_dir=args.prepared_dir, model_name=args.model, run_id=args.run_id, artifact_root=args.artifact_root, stages=stages, e1_run_root=args.e3_e1_run_root, e2_run_root=args.e3_e2_run_root, selected_heads=args.e3_head, max_tickers=args.max_tickers, grouping=args.e3_grouping)
        elif any(stage.startswith("e2-") for stage in stages):
            from llm_bias.entity_cell.e2 import run_e2
            root = run_e2(prepared_dir=args.prepared_dir, model_name=args.model, run_id=args.run_id, artifact_root=args.artifact_root, stages=stages, layers=args.e2_layers or (3, 7, 11, 15, 19, 23, 27, 31), max_tickers=args.max_tickers, lens_path=args.lens_path, expected_lens_sha256=args.expected_lens_sha256, expected_model_revision=args.expected_model_revision, expected_tokenizer_identity=args.expected_tokenizer_identity)
        else:
            from llm_bias.entity_cell.pipeline import run_e1
            root = run_e1(prepared_dir=args.prepared_dir, model_name=args.model, run_id=args.run_id, artifact_root=args.artifact_root, stages=stages, max_tickers=args.max_tickers)
    elif args.command == "analyze":
        if args.experiment == "discovery":
            if args.e1_run_root is None or args.e2_run_root is None or args.e3_run_root is None or args.output is None:
                raise ValueError("discovery analysis requires --e1-run-root, --e2-run-root, --e3-run-root, and --output")
            import json
            from llm_bias.entity_cell.confirmation import summarize_discovery
            from llm_bias.entity_cell.lifecycle import load_complete_run, require_stage
            values = {}
            for name, run_root, stage in (("e1", args.e1_run_root, "analyze"), ("e2", args.e2_run_root, "analyze"), ("e3", args.e3_run_root, "analyze")):
                _run, manifest = load_complete_run(run_root, label=f"{name.upper()} discovery run")
                require_stage(manifest, stage, label=f"{name.upper()} discovery run")
                values[name] = json.loads((run_root / "analyze" / "summary.json").read_text(encoding="utf-8"))
            from llm_bias.core.artifacts.io import write_json
            root = write_json(args.output, summarize_discovery(**values), overwrite=False)
        elif args.experiment == "e3":
            from llm_bias.entity_cell.e3 import analyze_e3
            root = analyze_e3(args.run_root)
        elif args.experiment == "e2":
            from llm_bias.entity_cell.e2 import analyze_e2
            root = analyze_e2(args.run_root)
        else:
            if args.run_root is None:
                raise ValueError("E1/E2/E3 analysis requires --run-root")
            from llm_bias.entity_cell.pipeline import analyze_e1
            root = analyze_e1(args.run_root)
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(root)


if __name__ == "__main__":
    main()
