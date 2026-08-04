import ast
import json
from pathlib import Path

from llm_bias.counterfactual_patching.cli import build_parser as patch_parser
from llm_bias.counterfactual_patching.data import default_spec_path
from llm_bias.counterfactual_patching.visualization import STATIC_DIR
from llm_bias.core.lens_artifacts import canonical_lens_path
from llm_bias.lens_fitting.calibration import load_calibration_prompts
from llm_bias.lens_fitting.cli import build_parser as lens_parser
from llm_bias.prompt_analysis.cli import build_parser as prompt_parser
from llm_bias.prompt_analysis.interactive import STATIC_DIR as PROMPT_STATIC_DIR


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

    assert set(patch_choices) == {
        "prepare-data",
        "run",
        "summarize",
        "visualize",
        "serve",
    }
    assert set(prompt_choices) == {
        "readout",
        "attribute",
        "validate-attribution",
        "visualize",
        "serve",
    }
    assert "fit-lens" not in patch_choices
    assert "fit-lens" not in prompt_choices
    attribute_args = prompt_parser().parse_args(
        [
            "attribute",
            "--model",
            "fake-qwen",
            "--runs",
            "30",
            "--temperature",
            "0.7",
            "--seed",
            "20260803",
            "--top-p",
            "1.0",
            "--top-k",
            "0",
        ]
    )
    assert attribute_args.runs == 30
    assert attribute_args.temperature == 0.7
    assert attribute_args.seed == 20260803
    assert attribute_args.top_p == 1.0
    assert attribute_args.top_k == 0
    lens_args = lens_parser().parse_args([])
    assert lens_args.output is None
    assert lens_args.layer_stride == 1
    assert lens_args.checkpoint_every == 4
    assert canonical_lens_path(lens_args.model) == Path(
        "artifacts/lenses/llama-3.2-1b-instruct/jacobian_lens.pt"
    )


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
