"""Deterministic contract tests for investment-dial calibration V2."""
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "investment_dial_calibration_v2",
    Path(__file__).parents[1] / "scripts/investment_dial_calibration_v2.py",
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def rows(split):
    return [
        {"id": f"{split}{i}", "ticker": f"T{i // 4}", "split": split, "positive_count": 2}
        for i in range(340)
    ]


def record(row, delta, *, layer=15, neuron=8490, decision="sell", phase=None, **extra):
    return dict(row, layer=layer, neuron=neuron, delta=delta, decision=decision,
                json_object=True, schema_valid=True, **({} if phase is None else {"phase": phase}), **extra)


def decision_rows(expected, delta, *, layer=15, neuron=8490, phase=None):
    # The fake response has a monotone, deterministic buy count as delta increases.
    buy_count = expected.get(float(delta), 170)
    return [record(row, delta, layer=layer, neuron=neuron,
                   decision="buy" if i < buy_count else "sell", phase=phase)
            for i, row in enumerate(rows("A"))]


def compact_bundle():
    a = rows("A")
    b = rows("B")
    coarse_values = {-8.0: -1.0, -4.0: -1.0, 4.0: 1.0, 8.0: 1.0}
    fine_values = {0.0: 0.0, 0.25: 0.4, 0.5: 0.7, 0.75: 0.85, 1.0: 0.92,
                   1.25: 0.96, 1.5: 0.98, 1.75: 0.99, 2.0: 1.0}
    def curve_records(population, delta, value, phase=None):
        buy = round(340 * (value + 1) / 2)
        return [record(row, delta, decision="buy" if i < buy else "sell", phase=phase)
                for i, row in enumerate(population)]
    return {
        "v1_protocol": {"runtime": {"torch": "t", "transformers": "f", "dtype": "d", "chat_template_sha256": "c"}},
        "fine_protocol": {"runtime": {"torch": "t", "transformers": "f", "dtype": "d", "chat_template_sha256": "c"}},
        "v1_manifest": {}, "fine_manifest": {}, "v1_digest": "v1", "fine_digest": "fine",
        "v1_a": a, "v1_b": b, "fine_a": a,
        "coarse": {delta: curve_records(a, delta, value, "A_curve") for delta, value in coarse_values.items()},
        "fine": {delta: curve_records(a, delta, value) for delta, value in fine_values.items()},
    }


def test_fixed_coordinates_population_and_record_identity_guards():
    with pytest.raises(ValueError, match="85"):
        m._validate_population(rows("A")[:-1], "A")
    with pytest.raises(ValueError, match="identities"):
        bad = rows("A")
        bad[-1]["id"] = bad[0]["id"]
        m._validate_population(bad, "A")
    with pytest.raises(ValueError, match="coordinate"):
        bad = [record(row, -0.5) for row in rows("A")]
        bad[0]["layer"] = 14
        m._validate_records(bad, rows("A"), -0.5)
    with pytest.raises(ValueError, match="phase"):
        m._validate_records([record(row, -0.5) for row in rows("A")], rows("A"), -0.5, require_phase="A_new")


def test_assembled_curve_is_exactly_15_points_and_nonextrapolating():
    bundle = compact_bundle()
    new = {-0.5: decision_rows({-0.5: 17}, -0.5, phase="A_new"),
           -0.25: decision_rows({-0.25: 51}, -0.25, phase="A_new")}
    curve = m.assemble_curve(bundle, new)
    assert [point["delta"] for point in curve["points"]] == list(m.ALL_DELTAS)
    assert curve["monotonicity"]["passed"]
    hats = m.inverse_curve([p["delta"] for p in curve["points"]],
                           [p["pi"] for p in curve["points"]], list(m.TARGETS))
    assert all(-8 <= value <= 8 for value in hats)
    with pytest.raises(ValueError, match="unreachable"):
        m.inverse_curve([-1, 0, 1], [-1, 0, 1], [1.1])


def test_invalid_nonmonotone_curve_and_missing_new_point_fail_closed():
    bundle = compact_bundle()
    new = {-0.5: decision_rows({-0.5: 200}, -0.5, phase="A_new"),
           -0.25: decision_rows({-0.25: 20}, -0.25, phase="A_new")}
    with pytest.raises(ValueError, match="non-decreasing"):
        m.assemble_curve(bundle, new)
    with pytest.raises(ValueError, match="delta"):
        m.assemble_curve(bundle, {-0.5: new[-0.5], -0.25: []})


def _patch_operator(monkeypatch, bundle, *, target_values=None):
    target_values = target_values or {-0.3: 119, 0.0: 170, 0.3: 221}
    monkeypatch.setattr(m, "_validate_v1_and_fine", lambda *args: bundle)
    monkeypatch.setattr(m.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(m.p, "load_model", lambda *args, **kwargs: (object(), object(), "cpu"))
    monkeypatch.setattr(m, "_verify_loaded_identity", lambda *args: None)
    monkeypatch.setattr(m.p, "frozen_eval", lambda *args: nullcontext())
    monkeypatch.setattr(m.p, "_identity", lambda *args: {"fake": True})
    monkeypatch.setattr(m.p, "_runtime", lambda *args: {"torch": "t", "transformers": "f", "dtype": "d", "chat_template_sha256": "c", "device": "cpu"})
    monkeypatch.setattr(m.p, "_source", lambda: {"fake": True})
    def fake_decisions(model, tokenizer, device, population, coordinate, delta, budget):
        phase = None
        if population is bundle["v1_a"] and delta in m.NEW_A_DELTAS:
            phase = "A_new"
        if population is bundle["v1_b"] and coordinate is None:
            phase = "B_baseline"
        if population is bundle["v1_b"] and coordinate is not None:
            phase = "B_reevaluation"
        if phase == "A_new":
            value = 17 if delta == -0.5 else 51
            buy = value
        elif phase == "B_baseline":
            buy = 170
        else:
            if target_values == {-0.3: 119, 0.0: 170, 0.3: 221}:
                buy = 119 if delta < -0.05 else 170 if delta < 0.1 else 221
            else:
                buy = target_values.get(round(delta, 12), 170)
        return [record(row, delta, layer=None if phase == "B_baseline" else 15,
                       neuron=None if phase == "B_baseline" else 8490,
                       decision="buy" if i < buy else "sell", phase=phase,
                       target=None if phase != "B_reevaluation" else 0.0,
                       delta_hat=delta if phase == "B_reevaluation" else None)
                for i, row in enumerate(population)]
    monkeypatch.setattr(m.p, "_decisions", fake_decisions)


def args(tmp_path, *, smoke=False, run_id="run"):
    return m.parser().parse_args([
        "--model", "fake-model", "--v1-run", "v1", "--fine-a-run", "fine",
        "--run-id", run_id, "--device", "cpu", "--artifact-root", str(tmp_path),
        *( ["--smoke"] if smoke else []),
    ])


def test_smoke_has_no_run_directory_and_does_not_invert(tmp_path, monkeypatch, capsys):
    bundle = compact_bundle()
    _patch_operator(monkeypatch, bundle)
    monkeypatch.setattr(m, "_smoke", lambda bundle, model, tokenizer, device: print("SMOKE TEST"))
    assert m.run(args(tmp_path, smoke=True)) is None
    assert not list(tmp_path.rglob("manifest.json"))
    assert "SMOKE TEST" in capsys.readouterr().out


def test_successful_formal_lifecycle_registers_outputs(tmp_path, monkeypatch):
    bundle = compact_bundle()
    _patch_operator(monkeypatch, bundle)
    directory = m.run(args(tmp_path, run_id="success"))
    manifest = json.loads((directory / "manifest.json").read_text())
    result = json.loads((directory / "analyze/result.json").read_text())
    assert manifest["status"] == "complete"
    assert result["gate_verdict"] == "pass"
    assert result["certified"] is True
    assert result["rmse"] == 0
    assert len(json.loads((directory / "forward/a_curve.json").read_text())["points"]) == 15
    assert len((directory / "forward/a-negative.jsonl").read_text().splitlines()) == 680
    assert len(json.loads((directory / "prepare/trials.json").read_text())) == 340


def test_gate_failure_and_existing_run_fail_before_model_load(tmp_path, monkeypatch):
    bundle = compact_bundle()
    _patch_operator(monkeypatch, bundle, target_values={-0.3: 170, 0.0: 170, 0.3: 170})
    directory = m.run(args(tmp_path, run_id="failure"))
    result = json.loads((directory / "analyze/result.json").read_text())
    assert result["gate_verdict"] == "fail" and not result["certified"]
    called = []
    monkeypatch.setattr(m.p, "load_model", lambda *a, **k: called.append(1))
    with pytest.raises(FileExistsError):
        m.run(args(tmp_path, run_id="failure"))
    assert not called


def test_parent_digest_and_identity_fail_closed(monkeypatch, tmp_path):
    with pytest.raises((FileNotFoundError, ValueError), match="missing|fixed parent manifest"):
        m._read_parent(tmp_path / "missing", "x", "0" * 64, set())
    bundle = compact_bundle()
    monkeypatch.setattr(m.p, "_identity", lambda *args: {"different": True})
    with pytest.raises(ValueError, match="identity"):
        m._verify_loaded_identity(bundle, "model", object(), object(), "cpu")


def test_cli_help():
    with pytest.raises(SystemExit) as exc:
        m.parser().parse_args(["--help"])
    assert exc.value.code == 0
