import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from llm_bias.counterfactual_patching.cli import build_parser as patch_parser
from llm_bias.counterfactual_patching.data import default_spec_path
from llm_bias.counterfactual_patching.visualization import STATIC_DIR
from llm_bias.core.lens_artifacts import canonical_lens_path
from llm_bias.lens_fitting.calibration import load_calibration_prompts
from llm_bias.lens_cli import build_parser as jacobian_lens_parser
from llm_bias.prompt_analysis import cli as prompt_cli
from llm_bias.prompt_analysis.cli import build_parser as prompt_parser
from llm_bias.prompt_analysis.interactive import STATIC_DIR as PROMPT_STATIC_DIR
from llm_bias.synthetic_entity_bias.cli import build_parser as synthetic_parser


def _llm_bias_imports(package: Path) -> set[str]:
    imports: set[str] = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("llm_bias."):
                    imports.add(node.module)
            elif isinstance(node, ast.Import):
                imports.update(
                    name.name
                    for name in node.names
                    if name.name.startswith("llm_bias.")
                )
    return imports


def test_runtime_packages_do_not_load_jacobian_lenses_directly():
    root = Path(__file__).resolve().parents[1] / "llm_bias"
    for package_name in (
        "counterfactual_patching",
        "prompt_analysis",
        "synthetic_entity_bias",
    ):
        for path in (root / package_name).glob("*.py"):
            assert "JacobianLens.load" not in path.read_text(encoding="utf-8"), path


def test_pretrained_lens_install_parser_is_explicit_and_offline_capable():
    parser = jacobian_lens_parser()
    args = parser.parse_args(
        [
            "install",
            "--model",
            ".cache/models/qwen3.5-4b",
            "--base-model",
            "Qwen/Qwen3.5-4B",
            "--offline",
            "--dry-run",
        ]
    )
    assert args.offline is True
    assert args.dry_run is True
    assert args.replace_existing is False


def test_experiment_packages_do_not_import_each_other():
    root = Path(__file__).resolve().parents[1] / "llm_bias"
    counterfactual_imports = _llm_bias_imports(root / "counterfactual_patching")
    prompt_imports = _llm_bias_imports(root / "prompt_analysis")

    assert not any(
        name.startswith("llm_bias.prompt_analysis")
        for name in counterfactual_imports
    )
    assert not any(
        name.startswith("llm_bias.counterfactual_patching")
        for name in prompt_imports
    )


def test_moved_package_resources_resolve_from_repository_root():
    assert default_spec_path().is_file()
    assert (STATIC_DIR / "counterfactual.html").is_file()
    assert (PROMPT_STATIC_DIR / "prompt_readout.html").is_file()


def test_independent_cli_command_sets():
    patch_choices = patch_parser()._subparsers._group_actions[0].choices
    prompt_choices = prompt_parser()._subparsers._group_actions[0].choices
    synthetic_choices = synthetic_parser()._subparsers._group_actions[0].choices

    assert set(patch_choices) == {
        "prepare-data",
        "run",
        "prepare-binary-association",
        "baseline-binary-association",
        "run-binary-patch",
        "fit-binary-direction",
        "run-binary-steering",
        "summarize-binary-association",
        "finalize-binary-manifest",
        "validate-binary-lens",
        "summarize",
        "visualize",
        "serve",
    }
    assert set(prompt_choices) == {
        "readout",
        "generate",
        "attribute-generated",
        "validate-attribution",
        "visualize",
        "plot-price-distributions",
        "plot-uncertainty-distributions",
        "inspect-input",
        "evaluate-return-predictions",
        "visualize-return-predictions",
        "serve",
    }
    assert set(synthetic_choices) == {"validate", "run", "visualize"}
    visualize_defaults = synthetic_parser().parse_args(
        ["visualize", "--run-root", "completed-run"]
    )
    assert visualize_defaults.run_root == "completed-run"
    assert visualize_defaults.output_dir is None
    assert visualize_defaults.replace_existing is False
    assert visualize_defaults.with_dashboard is False
    assert "fit-lens" not in patch_choices
    assert "fit-lens" not in prompt_choices
    readout_defaults = prompt_parser().parse_args(
        [
            "readout",
            "--model",
            "fake",
            "--lens",
            "fake-lens.pt",
            "--output-dir",
            "readout",
        ]
    )
    assert readout_defaults.input == "sp500_r1k_r2k_entityBiasPrompt.csv"
    assert readout_defaults.top_k == 15
    assert readout_defaults.batch_size == 32
    assert readout_defaults.max_seq_len == 256
    assert readout_defaults.use_chat_template is True
    assert readout_defaults.enable_thinking is False
    assert readout_defaults.save_prompt_topk is True
    assert readout_defaults.save_prompt_uncertainty is True
    with pytest.raises(SystemExit):
        prompt_parser().parse_args(
            [
                "readout",
                "--model",
                "fake",
                "--lens",
                "fake-lens.pt",
                "--output-dir",
                "readout",
                "--backprop",
            ]
        )

    generate_defaults = prompt_parser().parse_args(
        ["generate", "--model", "fake", "--output", "forward.jsonl"]
    )
    assert generate_defaults.input == "sp500_r1k_r2k_entityBiasPrompt.csv"
    assert generate_defaults.sample_per_condition == 32
    assert generate_defaults.max_new_tokens == 64
    assert generate_defaults.temperature == 0.0
    assert generate_defaults.top_p == 1.0
    assert generate_defaults.top_k == 0

    generate_args = prompt_parser().parse_args(
        [
            "generate",
            "--model",
            "fake-qwen",
            "--output",
            "forward.jsonl",
            "--sample-per-condition",
            "0",
            "--temperature",
            "0.7",
            "--seed",
            "20260803",
            "--runs",
            "2",
        ]
    )
    assert generate_args.sample_per_condition == 0
    assert generate_args.temperature == 0.7
    assert generate_args.seed == 20260803
    assert generate_args.runs == 2

    backward_defaults = prompt_parser().parse_args(
        [
            "attribute-generated",
            "--model",
            "fake",
            "--forward-artifact",
            "forward.jsonl",
            "--output",
            "backward.jsonl",
        ]
    )
    assert backward_defaults.max_seq_len == 256
    visualization_args = prompt_parser().parse_args(
        [
            "visualize",
            "--uncertainty-root",
            "run/readout",
            "--forward",
            "run/forward/generated_outputs.jsonl",
            "--backward",
            "run/backward/generated_token_attribution.jsonl",
            "--tokenizer",
            "fake-model",
            "--output-dir",
            "run/visualization",
        ]
    )
    assert visualization_args.forward.endswith("generated_outputs.jsonl")
    assert visualization_args.backward.endswith("generated_token_attribution.jsonl")
    return_evaluation_args = prompt_parser().parse_args(
        [
            "evaluate-return-predictions",
            "--forward",
            "run/forward/generated_outputs.jsonl",
            "--output-dir",
            "run/evaluation",
        ]
    )
    assert return_evaluation_args.forward.endswith("generated_outputs.jsonl")
    return_visualization_args = prompt_parser().parse_args(
        [
            "visualize-return-predictions",
            "--forward",
            "run/forward/generated_outputs.jsonl",
            "--uncertainty",
            "run/readout/prompt_layer_uncertainty.jsonl",
            "--output-dir",
            "run/visualization",
        ]
    )
    assert return_visualization_args.forward.endswith("generated_outputs.jsonl")
    price_plot_args = prompt_parser().parse_args(
        [
            "plot-price-distributions",
            "--sampling-root",
            "artifacts/sampling",
            "--prices",
            "prices.csv",
            "--output-dir",
            "artifacts/figures",
        ]
    )
    assert price_plot_args.sampling_root == "artifacts/sampling"
    assert price_plot_args.prices == "prices.csv"
    assert price_plot_args.output_dir == "artifacts/figures"
    uncertainty_plot_args = prompt_parser().parse_args(
        [
            "plot-uncertainty-distributions",
            "--uncertainty-root",
            "artifacts/readout",
            "--output-dir",
            "artifacts/uncertainty-figures",
        ]
    )
    assert uncertainty_plot_args.uncertainty_root == "artifacts/readout"
    assert uncertainty_plot_args.output_dir == "artifacts/uncertainty-figures"
    lens_args = jacobian_lens_parser().parse_args(["fit"])
    assert lens_args.output is None
    assert lens_args.layer_stride == 1
    assert lens_args.checkpoint_every == 4
    assert canonical_lens_path(lens_args.model) == Path(
        "artifacts/llama-3.2-1b-instruct/jacobian-lens/jacobian_lens.pt"
    )


def test_generated_attribution_fails_closed_without_forward_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-analysis",
            "attribute-generated",
            "--model",
            "fake",
            "--forward-artifact",
            str(tmp_path / "missing-forward.jsonl"),
            "--output",
            str(tmp_path / "backward.jsonl"),
        ],
    )

    with pytest.raises(FileNotFoundError, match="forward artifact"):
        prompt_cli.main()


def test_generation_cli_uses_forward_api_without_backward(monkeypatch, tmp_path):
    calls = []
    generation_module = ModuleType("llm_bias.prompt_analysis.generation")

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return Path(kwargs["output_path"])

    generation_module.generate_prompt_outputs = fake_generate
    monkeypatch.setitem(
        sys.modules, "llm_bias.prompt_analysis.generation", generation_module
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-analysis",
            "generate",
            "--model",
            "fake",
            "--input",
            "prompts.csv",
            "--output",
            str(tmp_path / "run" / "forward" / "generated_outputs.jsonl"),
            "--sample-per-condition",
            "0",
        ],
    )

    prompt_cli.main()

    assert len(calls) == 1
    assert calls[0]["full_generation"] is True
    assert calls[0]["sample_per_condition"] is None


def test_generated_attribution_cli_reads_forward_api_only(monkeypatch, tmp_path):
    forward = tmp_path / "forward.jsonl"
    forward.write_text("{}\n", encoding="utf-8")
    calls = []
    backward_module = ModuleType("llm_bias.prompt_analysis.generated_attribution")

    def fake_attribute(**kwargs):
        calls.append(kwargs)
        return Path(kwargs["output_path"])

    backward_module.attribute_generated_outputs = fake_attribute
    monkeypatch.setitem(
        sys.modules,
        "llm_bias.prompt_analysis.generated_attribution",
        backward_module,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-analysis",
            "attribute-generated",
            "--model",
            "fake",
            "--forward-artifact",
            str(forward),
            "--output",
            str(tmp_path / "run" / "backward" / "generated_token_attribution.jsonl"),
        ],
    )

    prompt_cli.main()

    assert len(calls) == 1
    assert calls[0]["forward_artifact"] == str(forward)
    assert calls[0]["output_path"] == tmp_path / "run" / "backward" / "generated_token_attribution.jsonl"
    assert "output_dir" not in calls[0]


def test_prompt_runner_manifest_records_canonical_refs_and_counts(tmp_path):
    root = Path(__file__).parents[1]
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("Date,prompt_without_context_a\n2026-01-01,hello\n", encoding="utf-8")
    lens = tmp_path / "lens.pt"
    lens.write_bytes(b"lens")
    artifact_root = tmp_path / "artifacts"
    run_root = artifact_root / "fake" / "data" / "runs" / "run"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
shift
if [[ $1 == python ]]; then shift; exec python "$@"; fi
if [[ $1 == prompt-analysis && $2 == generate ]]; then
  output=""
  while (($#)); do [[ $1 == --output ]] && output=$2; shift; done
  mkdir -p "$(dirname "$output")"
  printf '%s\\n' '{\"record_id\":\"r1\"}' > "$output"
  printf '%s\\n' '{}' > "$(dirname "$output")/metadata.json"
fi
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "MODEL": "fake", "LENS": str(lens), "INPUT_CSV": str(input_csv), "ARTIFACT_ROOT": str(artifact_root), "DATASET": "data", "RUN_ID": "run", "RUN_READOUT": "0", "RUN_GENERATION": "1", "RUN_ATTRIBUTION": "0", "RUN_IN_TMUX": "0"}
    result = subprocess.run(["bash", "scripts/run_prompt_analysis.sh"], cwd=root, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "complete"
    assert manifest["stages"]["generation"]["status"] == "complete"
    assert manifest["input_refs"][0]["sha256"] and manifest["lens_refs"][0]["sha256"]
    output = next(ref for ref in manifest["output_refs"] if ref["artifact_type"] == "generated_outputs")
    assert output["sha256"] and output["record_count"] == 1


def test_visualization_runner_passes_optional_validation_artifact():
    script = (
        Path(__file__).parents[1] / "scripts" / "visualize_prompt_analysis.sh"
    ).read_text(encoding="utf-8")

    assert "attribution_validation/semantic_scope_aopc.jsonl" in script
    assert "visualize_args+=(--validation \"${VALIDATION}\")" in script


def test_load_calibration_prompts_supports_text_and_jsonl(tmp_path):
    text_path = tmp_path / "prompts.txt"
    text_path.write_text("first prompt\n\nsecond prompt\n", encoding="utf-8")
    assert load_calibration_prompts(text_path) == ["first prompt", "second prompt"]

    jsonl_path = tmp_path / "prompts.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "one"}),
                json.dumps({"prompt": "two"}),
            ]
        ),
        encoding="utf-8",
    )
    assert load_calibration_prompts(jsonl_path, field="prompt", count=1) == ["one"]
