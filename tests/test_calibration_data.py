import json
from pathlib import Path


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_qwen_calibration_conditions_are_balanced_and_unique():
    root = Path(__file__).resolve().parents[1] / "data" / "calibration" / "qwen3.5-4b"
    english = _jsonl(root / "english.jsonl")
    chinese = _jsonl(root / "chinese_simplified.jsonl")
    mixed = _jsonl(root / "mixed.jsonl")

    assert len(english) == len(chinese) == len(mixed) == 128
    assert {row["language"] for row in english} == {"en"}
    assert {row["language"] for row in chinese} == {"zh-CN"}
    assert sum(row["language"] == "en" for row in mixed) == 64
    assert sum(row["language"] == "zh-CN" for row in mixed) == 64
    assert len({row["text"] for row in english}) == 128
    assert len({row["text"] for row in chinese}) == 128
    assert {row["pair_id"] for row in english} == {
        row["pair_id"] for row in chinese
    }
    for domain in {row["domain"] for row in mixed}:
        rows = [row for row in mixed if row["domain"] == domain]
        assert sum(row["language"] == "en" for row in rows) == 4
        assert sum(row["language"] == "zh-CN" for row in rows) == 4
    for style in {row["style"] for row in mixed}:
        rows = [row for row in mixed if row["style"] == style]
        assert sum(row["language"] == "en" for row in rows) == 8
        assert sum(row["language"] == "zh-CN" for row in rows) == 8


def test_bilingual_holdout_is_disjoint_and_paired():
    root = Path(__file__).resolve().parents[1]
    calibration_root = root / "data" / "calibration" / "qwen3.5-4b"
    holdout = _jsonl(
        root
        / "data"
        / "evaluations"
        / "qwen3.5-4b"
        / "bilingual_intermediate_holdout.jsonl"
    )
    calibration_text = {
        row["text"]
        for name in ("english", "chinese_simplified")
        for row in _jsonl(calibration_root / f"{name}.jsonl")
    }

    assert len(holdout) == 64
    assert len({row["id"] for row in holdout}) == 64
    assert sum(row["language"] == "en" for row in holdout) == 32
    assert sum(row["language"] == "zh-CN" for row in holdout) == 32
    assert all(row["prompt"] not in calibration_text for row in holdout)
    counts = {}
    for row in holdout:
        counts[row["pair_id"]] = counts.get(row["pair_id"], 0) + 1
    assert set(counts.values()) == {2}
