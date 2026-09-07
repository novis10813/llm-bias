"""Strict JSON decisions and explicit non-extrapolating calibration."""
import json
import math


def parse_response(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    def reject_constant(_value):
        raise ValueError("nonfinite JSON")
    try:
        obj = json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)
    except (ValueError, TypeError):
        return {"json_object": False, "decision": None, "schema_valid": False}
    is_object = isinstance(obj, dict)
    decision = obj.get("decision") if is_object else None
    if not isinstance(decision, str) or decision not in {"buy", "sell"}:
        decision = None
    return {"json_object": is_object, "decision": decision,
            "schema_valid": bool(decision is not None and isinstance(obj.get("reason"), str))}


def summary(rows):
    if not rows:
        raise ValueError("empty decision set")
    buy = sum(row["decision"] == "buy" for row in rows)
    sell = sum(row["decision"] == "sell" for row in rows)
    valid = buy + sell
    return {"n": len(rows), "buy": buy, "sell": sell,
            "pi": (buy - sell) / valid if valid else None,
            "valid_decision_rate": valid / len(rows),
            "parse_rate": sum(row["json_object"] for row in rows) / len(rows),
            "schema_rate": sum(row["schema_valid"] for row in rows) / len(rows)}


def inverse_curve(deltas, values, targets):
    if len(deltas) != len(values) or len(deltas) < 2:
        raise ValueError("invalid curve lengths")
    if any(value is None or not math.isfinite(value) for value in [*deltas, *values, *targets]):
        raise ValueError("nonfinite or undefined curve")
    if any(a >= b for a, b in zip(deltas, deltas[1:])):
        raise ValueError("deltas must increase")
    increasing = all(a <= b for a, b in zip(values, values[1:]))
    decreasing = all(a >= b for a, b in zip(values, values[1:]))
    if not (increasing or decreasing) or values[0] == values[-1]:
        raise ValueError("curve is not non-flat monotone")
    result = []
    for target in targets:
        if not min(values) <= target <= max(values):
            raise ValueError("target unreachable; extrapolation forbidden")
        exact = [d for d, v in zip(deltas, values) if v == target]
        if exact:
            result.append(min(exact, key=lambda d: (abs(d), d)))
            continue
        for d0, d1, v0, v1 in zip(deltas, deltas[1:], values, values[1:]):
            if min(v0, v1) < target < max(v0, v1):
                result.append(d0 + (d1 - d0) * (target - v0) / (v1 - v0))
                break
    return result


def feasible(stats, minimum_rate):
    return all(s["pi"] is not None and s["valid_decision_rate"] >= minimum_rate
               and s["schema_rate"] >= minimum_rate for s in stats)
