import json
from pathlib import Path

from llm_bias.counterfactual_patching.cli import _finish_manifest_stage, _manifest_for


def test_binary_manifest_registers_stage_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run-001"
    manifest_path = run_dir / "manifest.json"
    args = type("Args", (), {"manifest": str(manifest_path), "model": "test/model"})()
    manifest = _manifest_for(args)
    input_path = run_dir / "input.jsonl"
    output_path = run_dir / "output.jsonl"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text('{"id": "in"}\n', encoding="utf-8")
    output_path.write_text('{"id": "out"}\n', encoding="utf-8")
    _finish_manifest_stage(
        manifest,
        stage="baseline",
        inputs=[(input_path, "rendered_prompts")],
        outputs=[(output_path, "baseline_scores", "output", None)],
    )
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["status"] == "running"
    assert saved["stages"]["baseline"]["status"] == "complete"
    assert {item["artifact_type"] for item in saved["artifacts"]} == {
        "rendered_prompts",
        "baseline_scores",
    }
