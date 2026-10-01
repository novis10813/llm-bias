"""Outcome-only ITT accounting and issuer-cluster draws, without checkpoints."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import fields, replace
from fractions import Fraction

import pytest

from llm_bias.core import experiment_contract as C
from llm_bias.core.decision_metrics import (
    metrics_progress,
    paired_cluster_bootstrap,
    summarize_decisions,
)


FAILURES = (
    "exception", "invalid_json", "invalid_reason", "invalid_schema",
    "no_legal_token", "timeout", "truncated", "unsupported_channel",
)
ISSUERS = {"A1": "A", "A2": "A", "B": "B", "C": "C"}
GATES = {"no_op": True, "schema_supported": True}
ARMS = (("baseline", "0"), ("left", "1"), ("right", "2"))


def _hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _identity():
    return C.PlanIdentity(**{field.name: _hash(field.name) for field in fields(C.PlanIdentity)})


def _outcome(key, decision="buy", *, issuer=None, failure=None):
    finish = {"truncated": "token_budget", "timeout": "timeout", "exception": "exception",
              "no_legal_token": "no_legal_token", "unsupported_channel": "unsupported"}
    return C.GenerationOutcome(
        key.ticker, ISSUERS.get(key.ticker, key.ticker) if issuer is None else issuer,
        key.condition, key.trial_id, decision if failure is None else None,
        True, failure is None or failure == "invalid_reason", failure is None,
        finish.get(failure, "eos"), failure,
    )


@pytest.fixture
def cohort():
    # Per condition, seven buy-source/unknown units and one sell-source unit.
    # A1=1/1, A2=0/3 (including a failed intervention), B=1/2, C=unknown.
    units = [("A1", "1"), ("A2", "1"), ("A2", "2"), ("A2", "3"),
             ("B", "1"), ("B", "2"), ("B", "3"), ("C", "1")]
    keys = [C.RowKey("evaluation", ticker, condition, trial, arm, dose)
            for condition in ("+-", "-+") for ticker, trial in units for arm, dose in ARMS]
    plan = C.ExperimentPlan(_identity(), keys, tuple(GATES))
    rows = []
    for key in plan.keys:
        baseline = "sell" if (key.ticker, key.trial_id) == ("B", "3") else "buy"
        if key.condition == "-+":
            baseline = "buy" if baseline == "sell" else "sell"
        decision, failure = baseline, None
        if key.arm == "baseline" and key.ticker == "C":
            failure = "timeout"
        elif key.arm == "left":
            if key.ticker == "A1" or (key.ticker, key.trial_id) in (("B", "1"), ("B", "3")):
                decision = "sell" if baseline == "buy" else "buy"
            elif (key.ticker, key.trial_id) == ("A2", "3"):
                failure = "truncated"
        elif key.arm == "right":
            if key.ticker == "A2":
                decision = "sell" if baseline == "buy" else "buy"
            elif (key.ticker, key.trial_id) == ("B", "3"):
                failure = "invalid_reason"
        rows.append(C.ExecutionRow(key, _outcome(key, decision, failure=failure)))
    return plan, tuple(rows)


def _summary(cohort, **changes):
    plan, rows = cohort
    return summarize_decisions(plan, rows, **({"issuer_by_ticker": ISSUERS, "gates": GATES} | changes))


def _bootstrap(cohort, **changes):
    plan, rows = cohort
    return paired_cluster_bootstrap(plan, rows, **({
        "issuer_by_ticker": ISSUERS, "gates": GATES, "stage": "evaluation", "condition": "+-",
        "direction": "buy_to_sell", "left_arm": "left", "left_dose": "1",
        "right_arm": "right", "right_dose": "2", "seed": 7, "samples": 100,
        "confidence": 0.8,
    } | changes))


def _group(result, *, stage="evaluation", condition="+-", arm="left", dose="1"):
    return next(group for group in result["groups"] if
                (group["stage"], group["condition"], group["arm"], group["dose"]) ==
                (stage, condition, arm, dose))


def test_hand_company_first_and_fixed_denominator(cohort):
    result = _summary(cohort)
    assert set(result) == {"schema_version", "plan_hash", "complete", "eligible", "gates",
                           "issuer_mapping_sha256", "planned_rows", "executed_rows", "groups"}
    assert result["schema_version"] == 1
    assert result["plan_hash"] == cohort[0].plan_hash
    assert result["complete"] is result["eligible"] is True
    assert result["gates"] == GATES
    assert result["issuer_mapping_sha256"] == _hash(ISSUERS)
    assert result["planned_rows"] == result["executed_rows"] == 48
    assert len(result["groups"]) == 4
    group = _group(result)
    assert set(group) == {"stage", "condition", "arm", "dose", "planned_units", "planned_tickers",
                          "planned_issuers", "distinct_trial_ids", "baseline_buy", "baseline_sell",
                          "baseline_unknown", "baseline_primary_valid", "intervention_primary_valid",
                          "baseline_failures", "intervention_failures", "observed_flips",
                          "observed_flips_over_planned", "directions"}
    assert group["planned_units"] == 8  # Not the three distinct trial-id strings.
    assert group["planned_tickers"] == 4
    assert group["planned_issuers"] == 3
    assert group["distinct_trial_ids"] == 3
    assert (group["baseline_buy"], group["baseline_sell"], group["baseline_unknown"]) == (6, 1, 1)
    assert group["baseline_primary_valid"] == group["intervention_primary_valid"] == 7
    assert group["baseline_failures"] == dict.fromkeys(FAILURES, 0) | {"timeout": 1}
    assert group["intervention_failures"] == dict.fromkeys(FAILURES, 0) | {"truncated": 1}
    assert list(group["baseline_failures"]) == list(FAILURES)
    assert group["observed_flips"] == 3  # Unknown C with observed buy is not a flip.
    assert group["observed_flips_over_planned"] == 3 / 8
    direction = group["directions"]["buy_to_sell"]
    assert set(direction) == {"source_units", "flips", "retentions", "invalid_interventions",
                              "off_target_flips", "source_tickers", "source_free_tickers",
                              "company_first_rate", "micro_rate", "tickers"}
    assert direction["source_units"] == 6
    assert direction["flips"] == 2
    assert direction["retentions"] == 3
    assert direction["invalid_interventions"] == 1
    assert direction["off_target_flips"] == 1
    assert direction["source_tickers"] == 3
    assert direction["source_free_tickers"] == 1
    # Independent hand calculation: (1/1 + 0/3 + 1/2)/3 = 1/2; 2/(1+3+2) = 1/3.
    assert direction["company_first_rate"] == float((Fraction(1) + Fraction(0, 3) + Fraction(1, 2)) / 3)
    assert direction["micro_rate"] == float(Fraction(2, 6))
    assert direction["tickers"] == [
        {"ticker": "A1", "issuer_id": "A", "source_units": 1, "flips": 1, "rate": 1.0},
        {"ticker": "A2", "issuer_id": "A", "source_units": 3, "flips": 0, "rate": 0.0},
        {"ticker": "B", "issuer_id": "B", "source_units": 2, "flips": 1, "rate": 0.5},
        {"ticker": "C", "issuer_id": "C", "source_units": 0, "flips": 0, "rate": None},
    ]
    for group in result["groups"]:
        for direction in group["directions"].values():
            assert direction["flips"] + direction["retentions"] + direction["invalid_interventions"] == direction["source_units"]
    json.dumps(result, allow_nan=False)


def test_direction_condition_and_method_separation(cohort):
    result = _summary(cohort)
    forward = _group(result)["directions"]
    reverse = _group(result, condition="-+")["directions"]
    assert reverse["sell_to_buy"] == forward["buy_to_sell"]
    assert reverse["buy_to_sell"] == forward["sell_to_buy"]
    assert forward["sell_to_buy"]["source_units"] == forward["sell_to_buy"]["flips"] == 1
    assert forward["sell_to_buy"]["off_target_flips"] == 2
    right = _group(result, arm="right", dose="2")
    assert right["directions"]["buy_to_sell"]["company_first_rate"] == 1 / 3
    assert right["directions"]["buy_to_sell"]["micro_rate"] == 1 / 2
    assert right["directions"]["sell_to_buy"]["invalid_interventions"] == 1
    assert right["intervention_failures"]["invalid_reason"] == 1


def test_different_stage_populations_do_not_pair(cohort):
    plan, rows = cohort
    keys = [C.RowKey("localization", "C", "-+", "other-trial", arm, dose) for arm, dose in ARMS]
    extra = [C.ExecutionRow(key, _outcome(key, "sell" if key.arm == "left" else "buy")) for key in keys]
    extended = C.ExperimentPlan(plan.identity, [*plan.keys, *keys], tuple(GATES))
    result = _summary((extended, (*rows, *extra)))
    group = _group(result, stage="localization", condition="-+")
    assert group["planned_units"] == group["planned_tickers"] == group["planned_issuers"] == 1
    assert group["directions"]["buy_to_sell"]["company_first_rate"] == 1
    assert _group(result) == _group(_summary(cohort))
    assert _bootstrap((extended, (*rows, *extra)))["planned_issuers"] == 3


def test_progress_exact_shape_failures_and_incomplete_execution(cohort):
    plan, rows = cohort
    result = metrics_progress(plan, iter(rows))
    assert result == {
        "schema_version": 1, "plan_hash": plan.plan_hash, "planned": 48, "executed": 48,
        "missing": 0, "complete": True, "eligible": False,
        "gates": {name: None for name in sorted(GATES)},
        "generation_failures": dict.fromkeys(FAILURES, 0) | {"timeout": 2, "truncated": 2, "invalid_reason": 2},
    }
    assert metrics_progress(plan, rows, gates=GATES)["eligible"] is True
    partial = metrics_progress(plan, rows[:1], gates=GATES)
    assert (partial["executed"], partial["missing"], partial["complete"], partial["eligible"]) == (1, 47, False, False)
    assert list(partial["generation_failures"]) == list(FAILURES)
    assert metrics_progress(plan, (), gates=GATES)["generation_failures"] == dict.fromkeys(FAILURES, 0)
    failed_gate = metrics_progress(plan, rows, gates=GATES | {"no_op": False})
    assert failed_gate["complete"] and not failed_gate["eligible"]


@pytest.mark.parametrize("gate_state", [None, {}, {"no_op": True}, GATES | {"extra": True},
                                       GATES | {"no_op": 1}, GATES | {"no_op": None},
                                       GATES | {"no_op": "true"}, GATES | {"no_op": False}])
def test_efficacy_requires_exact_true_gates(cohort, gate_state):
    with pytest.raises(ValueError):
        _summary(cohort, gates=gate_state)
    with pytest.raises(ValueError):
        _bootstrap(cohort, gates=gate_state)
    if gate_state is not None and gate_state != GATES | {"no_op": False}:
        with pytest.raises(ValueError):
            metrics_progress(*cohort, gates=gate_state)


@pytest.mark.parametrize("mapping", [None, {}, {"A1": "A"}, ISSUERS | {"extra": "E"},
                                     ISSUERS | {"C": "wrong"}, ISSUERS | {"A1": " "},
                                     ISSUERS | {"B": 1}, {1: "A", "A2": "A", "B": "B", "C": "C"}])
def test_exact_nonblank_issuer_mapping_and_failed_outcome_metadata(cohort, mapping):
    with pytest.raises(ValueError):
        _summary(cohort, issuer_by_ticker=mapping)
    with pytest.raises(ValueError):
        _bootstrap(cohort, issuer_by_ticker=mapping)


@pytest.mark.parametrize("defect", ["missing_failed", "duplicate", "foreign", "wrong_issuer", "forged_outcome"])
def test_all_rows_validated_even_nonselected_groups(cohort, defect):
    plan, rows = cohort
    rows = list(rows)
    index = next(i for i, row in enumerate(rows) if row.key.condition == "-+" and row.key.arm == "right" and row.outcome.failure_type)
    if defect == "missing_failed":
        rows.pop(index)
    elif defect == "duplicate":
        rows.append(rows[index])
    elif defect == "foreign":
        key = replace(rows[index].key, stage="foreign")
        rows[index] = C.ExecutionRow(key, rows[index].outcome)
    elif defect == "wrong_issuer":
        rows[index] = C.ExecutionRow(rows[index].key, replace(rows[index].outcome, issuer_id="wrong"))
    else:
        object.__setattr__(rows[index].outcome, "decision", "buy")
    with pytest.raises(ValueError):
        _summary((plan, rows))
    with pytest.raises(ValueError):
        _bootstrap((plan, rows))
    if defect in ("duplicate", "foreign", "forged_outcome"):
        with pytest.raises(ValueError):
            metrics_progress(plan, rows)


@pytest.mark.parametrize("defect", ["wrong_baseline_dose", "missing_baseline", "missing_arm_unit", "extra_arm_unit", "baseline_only_stage"])
def test_entire_plan_alignment_not_observed_subset(cohort, defect):
    plan, rows = cohort
    rows = list(rows)
    if defect == "wrong_baseline_dose":
        rows = [C.ExecutionRow(replace(row.key, dose="1"), row.outcome)
                if row.key.arm == "baseline" and row.key.condition == "-+" else row for row in rows]
    elif defect == "missing_baseline":
        rows = [row for row in rows if row.key.arm != "baseline"]
    elif defect == "missing_arm_unit":
        rows.pop(next(i for i, row in enumerate(rows) if row.key.arm == "right" and row.key.condition == "-+"))
    elif defect == "extra_arm_unit":
        key = C.RowKey("evaluation", "C", "-+", "extra", "right", "2")
        rows.append(C.ExecutionRow(key, _outcome(key)))
    else:
        key = C.RowKey("baseline-only", "A1", "+-", "1", "baseline", "0")
        rows.append(C.ExecutionRow(key, _outcome(key)))
    changed = C.ExperimentPlan(plan.identity, [row.key for row in rows], tuple(GATES))
    assert metrics_progress(changed, rows, gates=GATES)["eligible"] is True
    with pytest.raises(ValueError):
        _summary((changed, rows))
    with pytest.raises(ValueError):
        _bootstrap((changed, rows))


@pytest.mark.parametrize("target", ["identity", "key", "gates", "duplicate_keys"])
def test_forged_plan_reconstruction(cohort, target):
    plan, rows = cohort
    if target == "identity":
        object.__setattr__(plan.identity, "model_sha256", "forged")
    elif target == "key":
        object.__setattr__(plan.keys[0], "dose", "0.00")
    elif target == "gates":
        object.__setattr__(plan, "gate_names", ("no_op", "no_op"))
    else:
        object.__setattr__(plan, "keys", (*plan.keys, plan.keys[0]))
    for call in (lambda: _summary((plan, rows)), lambda: _bootstrap((plan, rows)),
                 lambda: metrics_progress(plan, rows)):
        with pytest.raises(ValueError):
            call()


def test_row_plan_mapping_order_and_generator_inputs(cohort):
    plan, rows = cohort
    reordered = C.ExperimentPlan(plan.identity, reversed(plan.keys), reversed(plan.gate_names))
    mapping = dict(reversed(list(ISSUERS.items())))
    assert _summary((reordered, iter(reversed(rows))), issuer_by_ticker=mapping) == _summary(cohort)
    assert _bootstrap((reordered, iter(reversed(rows))), issuer_by_ticker=mapping) == _bootstrap(cohort)
    assert metrics_progress(reordered, iter(reversed(rows)), gates=GATES) == metrics_progress(plan, rows, gates=GATES)


def test_defensive_finite_json_outputs(cohort):
    before = (cohort[0].to_json(), tuple(row.to_json() for row in cohort[1]), dict(ISSUERS), dict(GATES))
    original = _summary(cohort)
    exported = _summary(cohort)
    exported["gates"]["no_op"] = False
    exported["groups"][0]["baseline_failures"]["timeout"] = 999
    exported["groups"][0]["directions"]["buy_to_sell"]["tickers"][0]["rate"] = 999
    assert _summary(cohort) == original
    boot = _bootstrap(cohort)
    json.dumps(boot, allow_nan=False)
    boot["left"]["interval"][0] = 999
    assert _bootstrap(cohort)["left"]["interval"][0] != 999
    state = metrics_progress(*cohort, gates=GATES)
    state["gates"].clear()
    state["generation_failures"].clear()
    assert before == (cohort[0].to_json(), tuple(row.to_json() for row in cohort[1]), ISSUERS, GATES)


# Independently enumerated issuer-occurrence counts. Each A contributes TWO ticker
# rates (left 1,0; right 0,1); B contributes ONE (left 1/2; right 0); C none.
# Thus left is always 1/2 in a finite draw, right=a/(2a+b), NOT an issuer mean.
DRAW_RATES = {
    (3, 0, 0): (Fraction(1, 2), Fraction(1, 2)),
    (2, 1, 0): (Fraction(1, 2), Fraction(2, 5)),
    (2, 0, 1): (Fraction(1, 2), Fraction(1, 2)),
    (1, 2, 0): (Fraction(1, 2), Fraction(1, 4)),
    (1, 1, 1): (Fraction(1, 2), Fraction(1, 3)),
    (1, 0, 2): (Fraction(1, 2), Fraction(1, 2)),
    (0, 3, 0): (Fraction(1, 2), Fraction(0)),
    (0, 2, 1): (Fraction(1, 2), Fraction(0)),
    (0, 1, 2): (Fraction(1, 2), Fraction(0)),
    (0, 0, 3): None,
}


def test_all_tiny_issuer_draws_include_share_classes_and_source_free(cohort, monkeypatch):
    draws = list(itertools.product(range(3), repeat=3))
    flat = iter(itertools.chain.from_iterable(draws))
    calls = []

    class EnumeratedRandom:
        def __init__(self, seed):
            assert seed == 7

        def randrange(self, count):
            calls.append(count)
            return next(flat)

    monkeypatch.setattr("llm_bias.core.decision_metrics.random.Random", EnumeratedRandom)
    result = _bootstrap(cohort, samples=27, confidence=0.8)
    assert calls == [3] * 81  # Source-free C is still in the three-issuer universe.
    assert result["planned_issuers"] == 3
    assert result["undefined_draws"] == 1
    assert result["finite_draws"] == 26
    assert result["left"]["estimate"] == 0.5
    assert result["right"]["estimate"] == 1 / 3
    assert result["difference"]["estimate"] == 0.5 - 1 / 3
    assert result["left"]["interval"] == [0.5, 0.5]
    # Among 26 finite draws: right=0 (7), 1/4 (3), 1/3 (6), 2/5 (3), 1/2 (7).
    # At q=.1/.9, positions=2.5/22.5: endpoints 0,1/2; paired diff also 0,1/2.
    assert result["right"]["interval"] == [0.0, 0.5]
    assert result["difference"]["interval"] == [0.0, 0.5]


def test_seeded_draw_order_against_hand_table(cohort):
    seed, samples, confidence = 6, 17, 0.7
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        draw = [rng.randrange(3) for _ in range(3)]
        values.append(DRAW_RATES[tuple(draw.count(i) for i in range(3))])
    finite = [pair for pair in values if pair is not None]
    assert any(pair is None for pair in values)  # Regression must exercise undefined draws.
    result = _bootstrap(cohort, seed=seed, samples=samples, confidence=confidence)
    assert result["undefined_draws"] == values.count(None)
    assert result["finite_draws"] == len(finite)
    assert set(result) == {"schema_version", "plan_hash", "issuer_mapping_sha256", "stage", "condition",
                           "direction", "left_arm", "left_dose", "right_arm", "right_dose",
                           "seed", "samples", "confidence", "planned_issuers", "finite_draws",
                           "undefined_draws", "left", "right", "difference"}
    assert result["schema_version"] == 1
    assert result["plan_hash"] == cohort[0].plan_hash
    assert result["issuer_mapping_sha256"] == _hash(ISSUERS)
    assert (result["seed"], result["samples"], result["confidence"]) == (seed, samples, confidence)
    for name, series in (
        ("left", [float(pair[0]) for pair in finite]),
        ("right", [float(pair[1]) for pair in finite]),
        ("difference", [float(pair[0]) - float(pair[1]) for pair in finite]),
    ):
        ordered = sorted(series)
        endpoints = []
        for q in ((1 - confidence) / 2, 1 - (1 - confidence) / 2):
            position = (len(ordered) - 1) * q
            lo = int(position)
            hi = min(lo + 1, len(ordered) - 1)
            endpoints.append(ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo]))
        assert result[name]["interval"] == pytest.approx(endpoints)
        assert set(result[name]) == {"estimate", "interval"}
    swapped = _bootstrap(cohort, seed=seed, samples=samples, confidence=confidence,
                         left_arm="right", left_dose="2", right_arm="left", right_dose="1")
    assert swapped["left"] == result["right"]
    assert swapped["right"] == result["left"]
    assert swapped["difference"]["estimate"] == -result["difference"]["estimate"]
    assert swapped["difference"]["interval"] == pytest.approx([-value for value in reversed(result["difference"]["interval"])])


def test_explicit_percentile_interpolation_and_paired_differences(cohort, monkeypatch):
    # ABC -> left .5/right 1/3, ABB -> .5/1/4, BBB -> .5/0, CCC undefined.
    sequence = iter([0, 1, 2, 0, 1, 1, 1, 1, 1, 2, 2, 2])

    class FixedRandom:
        def __init__(self, seed):
            pass

        def randrange(self, count):
            assert count == 3
            return next(sequence)

    monkeypatch.setattr("llm_bias.core.decision_metrics.random.Random", FixedRandom)
    result = _bootstrap(cohort, samples=4, confidence=0.5)
    assert result["finite_draws"] == 3 and result["undefined_draws"] == 1
    # Three finite draws, q=.25/.75 => positions=.5/1.5, average neighboring values.
    assert result["right"]["interval"] == pytest.approx([1 / 8, 7 / 24])
    assert result["difference"]["interval"] == pytest.approx([5 / 24, 3 / 8])


@pytest.mark.parametrize("changes", [
    {"seed": True}, {"seed": 1.0}, {"seed": "1"},
    {"samples": True}, {"samples": 0}, {"samples": -1}, {"samples": 2.0},
    {"confidence": True}, {"confidence": 0}, {"confidence": 1}, {"confidence": "0.9"},
    {"confidence": float("nan")}, {"confidence": float("inf")},
    {"direction": "buy"}, {"direction": []}, {"stage": "absent"},
    {"condition": "absent"}, {"stage": None}, {"left_arm": "baseline", "left_dose": "0"},
    {"left_dose": "1.0"}, {"left_dose": 1}, {"left_arm": "right", "left_dose": "2"},
    {"right_arm": "absent"}, {"right_dose": "3"},
])
def test_invalid_bootstrap_parameters(cohort, changes):
    with pytest.raises(ValueError):
        _bootstrap(cohort, **changes)


def _tiny(*, baseline="buy", shared=True, identical=False):
    mapping = {"A1": "only", "A2": "only" if shared else "other"}
    keys = [C.RowKey("evaluation", ticker, "+-", "1", arm, dose)
            for ticker in mapping for arm, dose in ARMS]
    plan = C.ExperimentPlan(_identity(), keys, tuple(GATES))
    rows = [C.ExecutionRow(key, _outcome(
        key, "sell" if key.arm == "left" or (identical and key.arm == "right") else baseline,
        issuer=mapping[key.ticker],
    )) for key in keys]
    return (plan, rows), mapping


def test_zero_source_class_null_not_nan():
    cohort, mapping = _tiny(baseline="sell")
    result = _summary(cohort, issuer_by_ticker=mapping)
    direction = _group(result)["directions"]["buy_to_sell"]
    assert direction["source_units"] == direction["source_tickers"] == 0
    assert direction["source_free_tickers"] == 2
    assert direction["company_first_rate"] is direction["micro_rate"] is None
    assert all(ticker["rate"] is None for ticker in direction["tickers"])
    boot = _bootstrap(cohort, issuer_by_ticker=mapping, samples=5)
    assert boot["undefined_draws"] == 5 and boot["finite_draws"] == 0
    assert boot["left"] == boot["right"] == boot["difference"] == {"estimate": None, "interval": None}
    json.dumps(boot, allow_nan=False)


def test_one_issuer_single_sample_and_identical_arms():
    cohort, mapping = _tiny()
    boot = _bootstrap(cohort, issuer_by_ticker=mapping, samples=1)
    assert boot["planned_issuers"] == boot["finite_draws"] == 1
    assert boot["undefined_draws"] == 0
    assert boot["left"] == {"estimate": 1.0, "interval": [1.0, 1.0]}
    assert boot["right"] == {"estimate": 0.0, "interval": [0.0, 0.0]}
    assert boot["difference"] == {"estimate": 1.0, "interval": [1.0, 1.0]}
    identical, mapping = _tiny(identical=True)
    boot = _bootstrap(identical, issuer_by_ticker=mapping, samples=9)
    assert boot["difference"] == {"estimate": 0.0, "interval": [0.0, 0.0]}


def test_bootstrap_direction_and_condition_selection(cohort):
    buy = _bootstrap(cohort)
    sell = _bootstrap(cohort, direction="sell_to_buy", condition="-+")
    for field in ("left", "right", "difference", "finite_draws", "undefined_draws"):
        assert buy[field] == sell[field]
    opposite = _bootstrap(cohort, direction="sell_to_buy")
    assert opposite["left"]["estimate"] == 1.0
    assert opposite["right"]["estimate"] == 0.0
    assert opposite["difference"]["interval"] == [1.0, 1.0]


@pytest.mark.parametrize("failure", FAILURES)
def test_all_failure_types_zero_defaults_and_fixed_denominator(failure):
    cohort, mapping = _tiny()
    plan, rows = cohort
    rows = [C.ExecutionRow(row.key, _outcome(row.key, issuer=mapping[row.key.ticker], failure=failure))
            if row.key.arm == "left" else row for row in rows]
    group = _group(_summary((plan, rows), issuer_by_ticker=mapping))
    direction = group["directions"]["buy_to_sell"]
    assert direction["source_units"] == direction["invalid_interventions"] == 2
    assert direction["flips"] == direction["retentions"] == 0
    assert direction["company_first_rate"] == direction["micro_rate"] == 0
    assert group["intervention_failures"] == dict.fromkeys(FAILURES, 0) | {failure: 2}
    state = metrics_progress(plan, rows, gates=GATES)
    assert state["generation_failures"] == dict.fromkeys(FAILURES, 0) | {failure: 2}


def test_removing_truncated_source_intervention_is_missing_not_exclusion(cohort):
    plan, rows = cohort
    missing = next(row for row in rows if row.key.condition == '+-' and row.key.arm == 'left'
                   and row.outcome.failure_type == 'truncated')
    remaining = [row for row in rows if row.key != missing.key]
    state = metrics_progress(plan, remaining, gates=GATES)
    assert state['missing'] == 1 and not state['eligible']
    with pytest.raises(ValueError, match='incomplete|missing'):
        _summary((plan, remaining))
    with pytest.raises(ValueError, match='incomplete|missing'):
        _bootstrap((plan, remaining))


def test_baseline_only_progress_is_permitted(cohort):
    plan, rows = cohort
    baseline = [row for row in rows if row.key.arm == 'baseline']
    baseline_plan = C.ExperimentPlan(plan.identity, [row.key for row in baseline], tuple(GATES))
    assert metrics_progress(baseline_plan, baseline, gates=GATES)['eligible'] is True
    assert metrics_progress(baseline_plan, [])['missing'] == 16
    with pytest.raises(ValueError, match='nonbaseline'):
        _summary((baseline_plan, baseline))


def test_all_undefined_draws_keep_existing_point_estimates(cohort, monkeypatch):
    class SourceFreeRandom:
        def __init__(self, seed):
            pass

        def randrange(self, count):
            assert count == 3
            return 2

    monkeypatch.setattr('llm_bias.core.decision_metrics.random.Random', SourceFreeRandom)
    result = _bootstrap(cohort, samples=1)
    assert result['undefined_draws'] == 1 and result['finite_draws'] == 0
    assert result['left'] == {'estimate': 0.5, 'interval': None}
    assert result['right'] == {'estimate': 1 / 3, 'interval': None}
    assert result['difference'] == {'estimate': 0.5 - 1 / 3, 'interval': None}


def test_group_doses_sort_as_canonical_strings(cohort):
    plan, rows = cohort
    # Same arm, two doses: lexical '10' precedes '2', unlike numeric sorting.
    changed = [C.ExecutionRow(replace(row.key, arm='steering', dose='10' if row.key.arm == 'left' else '2'),
                              row.outcome) if row.key.arm != 'baseline' else row for row in rows]
    new_plan = C.ExperimentPlan(plan.identity, [row.key for row in changed], tuple(GATES))
    result = _summary((new_plan, changed))
    assert [(g['condition'], g['arm'], g['dose']) for g in result['groups']] == [
        ('+-', 'steering', '10'), ('+-', 'steering', '2'),
        ('-+', 'steering', '10'), ('-+', 'steering', '2'),
    ]
