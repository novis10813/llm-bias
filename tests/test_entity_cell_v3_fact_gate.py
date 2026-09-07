"""Regression tests for the E1 V3 fact-level amnesia gate (fake model, no GPU)."""
import json

import pytest
import torch
from torch import nn

from llm_bias.entity_cell.analysis import v2_robustness_reasons
from llm_bias.entity_cell.cli import build_parser
from llm_bias.entity_cell.fact_amnesia import (
    FACT_FRAME_IDS,
    list_gold_decisions,
    read_verifications,
    run_fact_amnesia_stage,
    v3_candidate_eligibility,
    write_verifications,
)
from llm_bias.entity_cell.mlp_cells import OnlineVectorStats
from llm_bias.entity_cell.preparation import _prepare_fact_frames


class _Down(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, -1.0]]))

    def forward(self, value):
        return value @ self.weight.t()


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(2, 3, bias=False)
        self.up_proj = nn.Linear(2, 3, bias=False)
        self.down_proj = _Down()
        with torch.no_grad():
            self.gate_proj.weight.copy_(torch.ones(3, 2))
            self.up_proj.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))

    def forward(self, value):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(value)) * self.up_proj(value))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()

    def forward(self, value):
        return self.mlp(value)


class _Model(nn.Module):
    input_device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])

    def forward(self, input_ids):
        # O(1) logits so log-softmax stays well conditioned (no saturation).
        base = ((input_ids % 5).float() - 2.0).unsqueeze(-1).repeat(1, 1, 2) * 0.25
        mlp_out = self.layers[0](base)
        tail = torch.zeros(input_ids.shape[0], input_ids.shape[1], 1, device=base.device)
        return torch.cat([mlp_out, tail], dim=-1)


class _Tokenizer:
    def __call__(self, text, add_special_tokens=True, **_kwargs):
        return {"input_ids": [ord(char) % 97 for char in text]}

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(value % 26 + 97) for value in ids)


def _stats():
    stats = {0: OnlineVectorStats()}
    for value in (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 1.0])):
        stats[0].update(value)
    return stats


def _own_rows(ticker="AAA", cell=(0, 0), *, target=-1.0, wrong=-0.05, random_c=-0.05,
             degenerate=False, frame="F0"):
    rows = []
    for condition, logp, collapse in (
        ("clean", 0.0, None),
        ("target", target, target),
        ("wrong_entity", 0.0 if degenerate else wrong, None if degenerate else wrong),
        ("matched_random", random_c, random_c),
    ):
        rows.append({
            "check": "own", "ticker": ticker, "frame_id": frame, "prompt_id": f"{ticker}-{frame}",
            "candidate": {"layer": cell[0], "neuron": cell[1]},
            "wrong_entity_degenerate": degenerate,
            "gold_token_ids": [1, 2, 3], "gold_text": "abc", "gold_verified": None,
            "condition": condition, "gold_joint_logp": logp,
            "collapse": collapse,
        })
    return rows


def _all_own_rows(ticker, cell, target=-1.0):
    rows = []
    for frame in FACT_FRAME_IDS:
        rows.extend(_own_rows(ticker=ticker, cell=cell, target=target, frame=frame))
    return rows


def _cross_row(ticker, cell, other, collapse):
    return {
        "check": "cross", "ticker": ticker, "other_ticker": other, "frame_id": "F0",
        "prompt_id": f"{other}-F0", "candidate": {"layer": cell[0], "neuron": cell[1]},
        "gold_token_ids": [1, 2, 3], "gold_text": "def", "gold_verified": None,
        "wrong_entity_degenerate": False, "condition": "target",
        "gold_joint_logp": collapse, "collapse": collapse,
    }


def test_prepare_fact_frames_raw_text_and_span():
    class _OffsetTokenizer(_Tokenizer):
        def __call__(self, text, add_special_tokens=True, **kwargs):
            if kwargs.get("return_offsets_mapping") or kwargs.get("return_special_tokens_mask"):
                return {
                    "input_ids": [ord(char) % 500 for char in text],
                    "offset_mapping": [(index, index + 1) for index in range(len(text))],
                    "special_tokens_mask": [False] * len(text),
                }
            return super().__call__(text, add_special_tokens=add_special_tokens)

    source = [{"ticker": "AAA", "name": "Acme", "sector": "Technology", "source_row_index": 0}]
    rows = _prepare_fact_frames(_OffsetTokenizer(), source, split="discovery")
    assert len(rows) == 3
    assert [row["frame_id"] for row in rows] == list(FACT_FRAME_IDS)
    for row in rows:
        assert row["artifact_type"] == "entity_cell_fact_frame"
        assert row["prompt"].count("Acme") == 1
        assert row["prompt"] == row["prompt"].strip()
        assert len(row["input_ids"]) == len(row["prompt"])
        assert row["name_token_span"][1] > row["name_token_span"][0]
    with pytest.raises(ValueError, match="exactly once"):
        _prepare_fact_frames(_OffsetTokenizer(), [{"ticker": "AAA", "name": "", "sector": "Technology", "source_row_index": 0}], split="discovery")


def test_stage_emits_complete_rows_and_suppression_changes_logp():
    model = _Model()
    tokenizer = _Tokenizer()
    frames_by_ticker = {}
    for ticker in ("AAA", "BBB"):
        frames_by_ticker[ticker] = [
            {"frame_id": frame, "prompt": f"{frame} prompt for {ticker}", "prompt_id": f"{ticker}-{frame}"}
            for frame in FACT_FRAME_IDS
        ]
    cells_by_ticker = {
        "AAA": [{"layer": 0, "neuron": 0, "rank": 1}, {"layer": 0, "neuron": 1, "rank": 2}],
        "BBB": [{"layer": 0, "neuron": 1, "rank": 1}],
    }
    rows = run_fact_amnesia_stage(
        model=model, tokenizer=tokenizer, device=torch.device("cpu"), stats=_stats(),
        fact_frames_by_ticker=frames_by_ticker, cells_by_ticker=cells_by_ticker, tickers=["AAA", "BBB"],
    )
    own = [row for row in rows if row["check"] == "own"]
    cross = [row for row in rows if row["check"] == "cross"]
    assert len(own) == 2 * 3 * 4 + 1 * 3 * 4
    assert len(cross) == 2 * 1 + 1 * 1
    aaa_f0 = [row for row in own if row["ticker"] == "AAA" and row["frame_id"] == "F0" and row["candidate"] == {"layer": 0, "neuron": 0}]
    conditions = {row["condition"] for row in aaa_f0}
    assert conditions == {"clean", "target", "wrong_entity", "matched_random"}
    clean = next(row for row in aaa_f0 if row["condition"] == "clean")
    target = next(row for row in aaa_f0 if row["condition"] == "target")
    assert target["collapse"] == pytest.approx(target["gold_joint_logp"] - clean["gold_joint_logp"])
    assert target["collapse"] < 0.0
    assert clean["gold_verified"] is None
    # The matched-random neuron must differ from the target channel.
    random_rows = [row for row in aaa_f0 if row["condition"] == "matched_random"]
    assert random_rows and all(row["matched_random_neuron"] != 0 for row in random_rows)


def test_eligibility_classifications():
    cells = [{"layer": 0, "neuron": 0, "rank": 1}]
    rows = _all_own_rows("AAA", (0, 0))
    verified = {f"AAA:{frame}": {"verified": True} for frame in FACT_FRAME_IDS}

    entity = v3_candidate_eligibility(
        ticker="AAA", candidate_cells=cells, fact_rows=rows, verifications=verified, gates_1_3_pass=True,
    )
    assert entity["classification"] == "entity_cell"
    assert entity["trusted_cell"] == [0, 0]

    unrobust = v3_candidate_eligibility(
        ticker="AAA", candidate_cells=cells, fact_rows=rows, verifications=verified, gates_1_3_pass=False,
    )
    assert unrobust["classification"] == "fact_carrier_unrobust"
    assert unrobust["trusted_cell"] is None

    shared = v3_candidate_eligibility(
        ticker="AAA", candidate_cells=cells,
        fact_rows=rows + [_cross_row("AAA", (0, 0), "BBB", -0.8)],
        verifications={**verified, "BBB:F0": {"verified": True}}, gates_1_3_pass=False,
    )
    assert shared["classification"] == "shared_fact_channel"
    assert shared["channel_cells"] == [{"cell": [0, 0], "member_tickers": ["BBB"]}]

    no_verification = v3_candidate_eligibility(
        ticker="AAA", candidate_cells=cells, fact_rows=rows, verifications={}, gates_1_3_pass=True,
    )
    assert no_verification["classification"] == "fact_gate_not_applicable"

    small_effect = v3_candidate_eligibility(
        ticker="AAA", candidate_cells=cells, fact_rows=_all_own_rows("AAA", (0, 0), target=-0.2),
        verifications=verified, gates_1_3_pass=True,
    )
    assert small_effect["classification"] == "not_eligible"

    no_rows = v3_candidate_eligibility(
        ticker="AAA", candidate_cells=cells, fact_rows=[], verifications=verified, gates_1_3_pass=True,
    )
    assert no_rows["classification"] == "fact_gate_unavailable"


def test_primary_classification_ignores_lower_rank_shared_channels():
    # JPM pattern: top-1 passes on its own (Gates 1-3 fail); a lower-ranked
    # candidate is a shared channel. Primary state = top-1, channel reported.
    cells = [{"layer": 0, "neuron": 0, "rank": 1}, {"layer": 0, "neuron": 1, "rank": 2}]
    verified = {f"AAA:{frame}": {"verified": True} for frame in FACT_FRAME_IDS}
    rows = _all_own_rows("AAA", (0, 0)) + _all_own_rows("AAA", (0, 1))
    rows.append(_cross_row("AAA", (0, 1), "BBB", -0.8))
    result = v3_candidate_eligibility(
        ticker="AAA", candidate_cells=cells, fact_rows=rows,
        verifications={**verified, "BBB:F0": {"verified": True}}, gates_1_3_pass=False,
    )
    assert result["classification"] == "fact_carrier_unrobust"
    assert result["trusted_cell"] is None
    assert result["channel_cells"] == [{"cell": [0, 1], "member_tickers": ["BBB"]}]
    assert result["passing_candidates"] == ["0:0", "0:1"]

    # Same geometry with Gates 1-3 passing: the private top-1 is the entity cell.
    result = v3_candidate_eligibility(
        ticker="AAA", candidate_cells=cells, fact_rows=rows,
        verifications={**verified, "BBB:F0": {"verified": True}}, gates_1_3_pass=True,
    )
    assert result["classification"] == "entity_cell"
    assert result["trusted_cell"] == [0, 0]
    assert result["trusted_cell_rank"] == 1


def test_specificity_requires_flat_controls_and_degenerate_fallback():
    cells = [{"layer": 0, "neuron": 0, "rank": 1}]
    noisy = _all_own_rows("AAA", (0, 0))
    for row in noisy:
        if row["condition"] == "matched_random":
            row["collapse"] = -0.4
    verified = {f"AAA:{frame}": {"verified": True} for frame in FACT_FRAME_IDS}
    result = v3_candidate_eligibility(
        ticker="AAA", candidate_cells=cells, fact_rows=noisy, verifications=verified, gates_1_3_pass=True,
    )
    assert result["classification"] == "not_eligible"

    degenerate = _all_own_rows("AAA", (0, 0))
    for row in degenerate:
        if row["condition"] == "wrong_entity":
            row["wrong_entity_degenerate"] = True
            row["collapse"] = None
    result = v3_candidate_eligibility(
        ticker="AAA", candidate_cells=cells, fact_rows=degenerate, verifications=verified, gates_1_3_pass=True,
    )
    assert result["classification"] == "entity_cell"


def test_verification_roundtrip_and_fail_closed(tmp_path):
    path = tmp_path / "fact_gold_verifications.json"
    write_verifications(path, {"AAA:F0": {"verified": True, "note": "correct HQ"}})
    payload = json.loads(path.read_text())
    assert payload["artifact_type"] == "entity_cell_fact_gold_verifications"
    assert read_verifications(path) == {"AAA:F0": {"verified": True, "note": "correct HQ"}}
    path.write_text(json.dumps({"decisions": {"AAA:F0": {"verified": "yes"}}}))
    with pytest.raises(ValueError, match="boolean"):
        read_verifications(path)


def test_list_gold_decisions_deduplicates_own_and_cross():
    rows = _all_own_rows("AAA", (0, 0))
    rows.append(_cross_row("AAA", (0, 0), "BBB", -0.8))
    keys = [row["key"] for row in list_gold_decisions(rows)]
    assert keys == sorted(keys)
    assert "AAA:F0" in keys and "BBB:F0" in keys
    assert len(keys) == len(set(keys))


def test_cli_accepts_fact_stage_and_verify_command():
    parser = build_parser()
    args = parser.parse_args([
        "run-localization", "--prepared-dir", "p", "--model", "m", "--run-id", "r",
        "--stage", "e1-fact-amnesia",
    ])
    assert args.stages == ["e1-fact-amnesia"]
    args = parser.parse_args(["verify-fact-gold", "--run-dir", "run", "--list"])
    assert args.command == "verify-fact-gold" and args.list_rows


def test_v2_robustness_reasons_match_v2_gate_semantics():
    signature = [{"layer": 0, "neuron": 9}]
    reasons = v2_robustness_reasons(
        held_metrics={"top5_overlap": 0},
        surface_control_summary={"form_robust": False},
        candidate_cells=[{"layer": 0, "neuron": 9}],
        template_signature=signature,
    )
    assert reasons == ["held_variant_top5_overlap_zero", "not_form_robust", "in_template_signature"]
