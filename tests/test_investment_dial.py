import json
import re
from types import SimpleNamespace

import pytest
import torch

from llm_bias.investment_dial import pipeline
from llm_bias.investment_dial.analysis import parse_response, summary, inverse_curve
from llm_bias.investment_dial.prompts import build_trials, encode_trials, validate_data
from llm_bias.core.inference.coordinate_screen import coordinate_derivatives
from llm_bias.core.artifacts.registered import verified_run


def data():
    return {"schema_version": 1, "source": "synthetic unit-test evidence, NOT baseline",
            "companies": [{"ticker": split, "name": f"Test {split}", "split": split,
                           "evidence_pairs": [{"positive": f"Positive {i}", "negative": f"Negative {i}"} for i in range(4)]}
                          for split in ("screen", "A", "B", "test")]}


class Tokenizer:
    chat_template = "fake"
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "USER:" + messages[0]["content"] + "ASSISTANT:"

    def __call__(self, text, **kwargs):
        tokens = re.findall(r"buy|sell|[\s\S]", text)
        return {"input_ids": [2 if t == "buy" else 3 if t == "sell" else ord(t) + 4 for t in tokens]}


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.mlp.down_proj.weight.zero_()
            self.mlp.down_proj.weight[2, 0] = 1
            self.mlp.down_proj.weight[3, 1] = .1


class Raw(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer()])

    def forward(self, input_ids, **kwargs):
        x = torch.ones(*input_ids.shape, 3)
        return SimpleNamespace(logits=self.layers[0].mlp.down_proj(x).cumsum(1))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data()))
    raw = Raw()
    model = SimpleNamespace(_hf_model=raw, layers=raw.layers)
    loaded = model, Tokenizer(), torch.device("cpu")
    monkeypatch.setattr(pipeline, "_identity", lambda *_: {"fake": True})
    seen = []

    def generate(model, tokenizer, device, row, budget):
        seen.append(row["split"])
        probe = model.layers[0].mlp.down_proj(torch.zeros(1, 1, 3))[0, 0]
        offset = .5 if row["reverse_options"] else -.5
        decision = "buy" if float(probe[2] - probe[3]) + offset > 0 else "sell"
        text = json.dumps({"decision": decision, "reason": "fake only"})
        return {k: row[k] for k in ("id", "ticker", "split", "positive_count", "reverse_options")} | {
            "text": text, "generated_ids": [2 if decision == "buy" else 3], "finish_reason": "eos_token"
        } | parse_response(text)

    monkeypatch.setattr(pipeline, "_generate", generate)
    return path, loaded, seen


def test_prompts_balanced_options_and_splits():
    rows = build_trials(data(), repeats=1)
    assert len(rows) == 8
    assert build_trials(data(), repeats=1) == rows
    for first, second in zip(rows[::2], rows[1::2]):
        assert first["ticker"] == second["ticker"]
        assert first["prompt"].count("- Positive") == 2
        assert first["prompt"].count("- Negative") == 2
        assert not first["reverse_options"] and second["reverse_options"]
    assert all(len(row["answer_ids"]) == 2 for row in encode_trials(rows, Tokenizer()))
    bad = data()
    bad["companies"][1]["ticker"] = "screen"
    with pytest.raises(ValueError, match="duplicate"):
        validate_data(bad)


@pytest.mark.parametrize("text", ['not json', '[]', '{"decision":"hold","reason":"x"}',
    '{"decision":"buy","decision":"sell","reason":"x"}',
    '```json\n{"decision":"buy","reason":"x"}\n```', '{"decision":[]}'])
def test_invalid_decisions_are_not_guessed(text):
    assert parse_response(text)["decision"] is None


def test_parse_rates_and_undefined_pi():
    missing_reason = parse_response('{"decision":"buy"}')
    assert missing_reason["decision"] == "buy" and not missing_reason["schema_valid"]
    stats = summary([missing_reason, parse_response('bad')])
    assert stats["pi"] == 1 and stats["valid_decision_rate"] == .5 and stats["schema_rate"] == 0
    assert summary([parse_response('bad')])["pi"] is None


def test_inverse_branches_and_rejection():
    assert inverse_curve([-2, 0, 2], [-1, 0, 1], [-.5, 0, .5]) == [-1, 0, 1]
    assert inverse_curve([-2, 0, 2], [1, 0, -1], [-.5, 0, .5]) == [1, 0, -1]
    assert inverse_curve([-2, 0, 2, 4], [-1, 0, 0, 1], [0]) == [0]
    for values in ([-1, 1, 0], [0, 0, 0], [None, 0, 1]):
        with pytest.raises(ValueError):
            inverse_curve([-2, 0, 2], values, [0])
    with pytest.raises(ValueError, match="unreachable"):
        inverse_curve([-2, 0, 2], [-.1, 0, .1], [.5])


def test_screen_calibrate_evaluate_lifecycle(setup, tmp_path):
    path, loaded, seen = setup
    root = tmp_path / "artifacts"
    screen = pipeline.run_screen(path, "fake", "screen", artifact_root=root, repeats=1, top_k=2, loaded=loaded)
    result = json.loads((screen / "analyze/result.json").read_text())
    assert result["candidates"][0]["neuron"] == 0
    assert result["candidates"][1]["signed_sensitivity"] < 0
    assert not seen
    calibrated = pipeline.run_calibration(screen, "fake", "cal", artifact_root=root, loaded=loaded)
    assert set(seen) == {"A", "B"}
    seen.clear()
    evaluated = pipeline.run_evaluation(calibrated, "fake", "eval", artifact_root=root, loaded=loaded)
    assert set(seen) == {"test"}
    output, _ = verified_run(evaluated, "investment-dial-evaluation", pipeline.REQUIRED)
    assert len(output["analyze/result.json"]["summaries"]) == 35
    assert not output["analyze/result.json"]["certified"]
    assert output["prepare/protocol.json"]["control_neuron"] == 2
    for directory in (screen, calibrated, evaluated):
        assert json.loads((directory / "manifest.json").read_text())["status"] == "complete"
    assert not loaded[0].layers[0].mlp.down_proj._forward_pre_hooks
    assert all(p.requires_grad for p in loaded[0]._hf_model.parameters())


def test_no_feasible_candidate_is_complete_not_certified(setup, tmp_path):
    path, loaded, _ = setup
    screen = pipeline.run_screen(path, "fake", "s", artifact_root=tmp_path, top_k=1, loaded=loaded)
    cal = pipeline.run_calibration(screen, "fake", "c", artifact_root=tmp_path,
                                   deltas=[-.01, 0, .01], loaded=loaded)
    result = json.loads((cal / "analyze/result.json").read_text())
    assert result["selected"] is None and not result["success"]
    with pytest.raises(ValueError, match="no feasible"):
        pipeline.run_evaluation(cal, "fake", "e", artifact_root=tmp_path, loaded=loaded)


def test_tamper_and_model_mismatch_rejected(setup, tmp_path, monkeypatch):
    path, loaded, _ = setup
    screen = pipeline.run_screen(path, "fake", "s", artifact_root=tmp_path, loaded=loaded)
    monkeypatch.setattr(pipeline, "_identity", lambda *_: {"different": True})
    with pytest.raises(ValueError, match="model/tokenizer"):
        pipeline.run_calibration(screen, "fake", "c", artifact_root=tmp_path, loaded=loaded)
    monkeypatch.setattr(pipeline, "_identity", lambda *_: {"fake": True})
    (screen / "analyze/result.json").write_text('{}')
    with pytest.raises(ValueError, match="hash mismatch"):
        pipeline.run_calibration(screen, "fake", "c", artifact_root=tmp_path, loaded=loaded)


def test_heldout_changes_cannot_change_screen(setup, tmp_path):
    path, loaded, _ = setup
    a = pipeline.run_screen(path, "fake", "a", artifact_root=tmp_path, loaded=loaded)
    changed = data()
    changed["companies"][-1]["evidence_pairs"][0]["positive"] = "UNTOUCHED DIFFERENT EVIDENCE"
    path.write_text(json.dumps(changed))
    b = pipeline.run_screen(path, "fake", "b", artifact_root=tmp_path, loaded=loaded)
    assert json.loads((a / "analyze/result.json").read_text())["candidates"] == json.loads((b / "analyze/result.json").read_text())["candidates"]


def test_signed_ticker_means_cancel_before_absolute_value(setup, tmp_path, monkeypatch):
    path, loaded, _ = setup
    inputs = data()
    extra = dict(inputs["companies"][0], ticker="screen2", name="Another screen company")
    inputs["companies"].append(extra)
    path.write_text(json.dumps(inputs))
    calls = []
    def derivatives(*args):
        calls.append(1)
        return 0., {0: torch.tensor([1. if len(calls) <= 2 else -1., .25])}
    monkeypatch.setattr(pipeline, "coordinate_derivatives", derivatives)
    run = pipeline.run_screen(path, "fake", "cancel", artifact_root=tmp_path, repeats=1, top_k=1, loaded=loaded)
    candidate = json.loads((run / "analyze/result.json").read_text())["candidates"][0]
    assert candidate["neuron"] == 1 and candidate["signed_sensitivity"] == .25


def test_engineering_check_lifecycle(setup, tmp_path):
    path, loaded, _ = setup
    run = pipeline.run_check(path, "fake", "check", artifact_root=tmp_path, loaded=loaded)
    result = json.loads((run / "analyze/result.json").read_text())
    assert result["numeric_agreement"] and result["zero_identical"]
    assert result["scope"] == "single_prompt_engineering_only"
    assert not result["certified"]


def test_failure_marks_manifest_and_removes_hooks(setup, tmp_path, monkeypatch):
    path, loaded, _ = setup
    screen = pipeline.run_screen(path, "fake", "s", artifact_root=tmp_path, loaded=loaded)
    def explode(*args):
        raise RuntimeError("generation failure")
    monkeypatch.setattr(pipeline, "_generate", explode)
    with pytest.raises(RuntimeError, match="generation failure"):
        pipeline.run_calibration(screen, "fake", "broken", artifact_root=tmp_path, loaded=loaded)
    manifests = [json.loads(p.read_text()) for p in tmp_path.rglob("manifest.json")]
    assert any(m["status"] == "failed" for m in manifests)
    assert not loaded[0].layers[0].mlp.down_proj._forward_pre_hooks
    assert all(p.requires_grad for p in loaded[0]._hf_model.parameters())


def test_missing_input_fails_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_model", lambda *_: pytest.fail("should not load weights"))
    with pytest.raises(FileNotFoundError):
        pipeline.run_screen(tmp_path / "missing.json", "fake", "s")


def test_heldout_sampling_independent_of_screen():
    original = data()
    before = [r for r in build_trials(original) if r["split"] == "screen"]
    original["companies"][-1]["ticker"] = "000"
    original["companies"][-1]["evidence_pairs"] *= 3
    after = [r for r in build_trials(original) if r["split"] == "screen"]
    assert before == after


def test_multitoken_answer_rejected():
    class CharacterTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) for c in text]}
    with pytest.raises(ValueError, match="single-token"):
        encode_trials(build_trials(data()), CharacterTokenizer())


def test_gradient_flags_restored_after_exception():
    raw = Raw()
    model = SimpleNamespace(_hf_model=raw, layers=raw.layers)
    with pytest.raises(IndexError):
        coordinate_derivatives(model, [1, 2], 500, 3, "cpu", [0])
    assert raw.training and all(p.requires_grad for p in raw.parameters())
    assert not raw.layers[0].mlp.down_proj._forward_pre_hooks


def test_cli_dispatch(monkeypatch):
    from llm_bias.investment_dial import cli
    seen = {}
    monkeypatch.setattr(cli, "run_screen", lambda **kwargs: seen.update(kwargs))
    cli.main(["run-screen", "--input", "input.json", "--run-id", "test", "--layers", "0", "2"])
    assert seen["layers"] == [0, 2] and seen["input_path"] == "input.json"
