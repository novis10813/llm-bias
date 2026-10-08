"""Synthetic, checkpoint-free acceptance for the native no-tool boundary protocol."""
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
import subprocess
import sys

import pytest

from llm_bias.core.inference.harmony_channels import (
    HarmonyBoundaryTracker,
    HarmonyTokenContract,
    UnsupportedHarmonyChannel,
)

# Explicit synthetic bindings, including multi-token role/channel names and a
# shared header prefix. Ordinary names may repeat; these are not checkpoint IDs.
START, CHANNEL, MESSAGE, END, STOP, TOOL, SCHEMA, UNKNOWN = range(100, 108)
RESTART = (START, 21, 22)
ANALYSIS = (CHANNEL, 11, 12, 11, MESSAGE)
FINAL = (CHANNEL, 11, 13, MESSAGE)
SPECIALS = (START, CHANNEL, MESSAGE, END, STOP, TOOL, SCHEMA, UNKNOWN)
CONTRACT = HarmonyTokenContract(
    prompt_suffix_ids=RESTART,
    initial_analysis_header_ids=ANALYSIS,
    initial_final_header_ids=FINAL,
    message_end_id=END,
    assistant_restart_ids=RESTART,
    final_stop_id=STOP,
    forbidden_control_ids=(TOOL, SCHEMA),
)
# The first body can stand for ordinary words "final" and fake JSON. No text
# lookup is permitted. The second analysis is empty, but still completed.
REPEATED = ANALYSIS + (11, 13, 30, 31, END) + RESTART + ANALYSIS + (END,) + RESTART + FINAL + (32, 33, 34, STOP)
DIRECT = FINAL + (32, 33, 34, STOP)


def tracker():
    return HarmonyBoundaryTracker(CONTRACT, special_token_ids=SPECIALS)


def state(item):
    return (item.phase, item.final_content_start, item.final_content_end,
            item.analysis_segments, item.observed_token_ids, item.is_terminated,
            item.allowed_next_ids())


def test_contract_fields_and_defensive_tuple_binding():
    assert {field.name for field in fields(CONTRACT)} == {
        "prompt_suffix_ids", "initial_analysis_header_ids", "initial_final_header_ids",
        "message_end_id", "assistant_restart_ids", "final_stop_id", "forbidden_control_ids",
    }
    changes = {name: list(getattr(CONTRACT, name)) for name in (
        "prompt_suffix_ids", "initial_analysis_header_ids", "initial_final_header_ids",
        "assistant_restart_ids", "forbidden_control_ids",
    )}
    copied = replace(CONTRACT, **changes)
    for values in changes.values():
        values.clear()
    assert copied == CONTRACT
    assert hash(copied) == hash(CONTRACT)
    assert not hasattr(copied, "__dict__")
    for field in fields(CONTRACT):
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(copied, field.name, getattr(CONTRACT, field.name))
    assert replace(CONTRACT, forbidden_control_ids=())
    assert replace(CONTRACT, assistant_restart_ids=iter(RESTART)) == CONTRACT


class _IntSubclass(int):
    pass


BAD_SEQUENCES = ("123", b"123", bytearray(b"123"), {1: 2}, None, 123,
                 [True], [False], [-1], [1.0], ["1"], [None], [[1]], [_IntSubclass(1)])


@pytest.mark.parametrize("field", [
    "prompt_suffix_ids", "initial_analysis_header_ids", "initial_final_header_ids",
    "assistant_restart_ids", "forbidden_control_ids",
])
@pytest.mark.parametrize("value", BAD_SEQUENCES)
def test_contract_rejects_invalid_sequences(field, value):
    with pytest.raises(ValueError):
        replace(CONTRACT, **{field: value})


@pytest.mark.parametrize("field", ["message_end_id", "final_stop_id"])
@pytest.mark.parametrize("value", [True, False, -1, 1.0, "1", None, [], {}, _IntSubclass(1)])
def test_contract_rejects_invalid_scalar_ids(field, value):
    with pytest.raises(ValueError):
        replace(CONTRACT, **{field: value})


@pytest.mark.parametrize("changes", [
    {"prompt_suffix_ids": ()}, {"initial_analysis_header_ids": ()},
    {"initial_final_header_ids": ()}, {"assistant_restart_ids": ()},
    {"initial_analysis_header_ids": (CHANNEL, MESSAGE)},
    {"initial_final_header_ids": (CHANNEL, MESSAGE)},
    {"assistant_restart_ids": (START,), "prompt_suffix_ids": (START,)},
    {"prompt_suffix_ids": (START, 21)},
    {"initial_final_header_ids": ANALYSIS},
    {"initial_analysis_header_ids": (CHANNEL, 11, MESSAGE),
     "initial_final_header_ids": (CHANNEL, 11, MESSAGE, 12, MESSAGE)},
    {"initial_analysis_header_ids": (CHANNEL, 11, MESSAGE, 12, MESSAGE),
     "initial_final_header_ids": (CHANNEL, 11, MESSAGE)},
    {"initial_final_header_ids": (99, 11, 13, MESSAGE)},
    {"initial_final_header_ids": (CHANNEL, 11, 13, 99)},
    {"message_end_id": STOP}, {"message_end_id": CHANNEL},
    {"message_end_id": MESSAGE}, {"message_end_id": START},
    {"final_stop_id": CHANNEL}, {"final_stop_id": MESSAGE},
    {"final_stop_id": START},
    {"assistant_restart_ids": (CHANNEL, 21), "prompt_suffix_ids": (CHANNEL, 21)},
    {"initial_analysis_header_ids": (CHANNEL, 11, START, MESSAGE)},
    {"initial_final_header_ids": (CHANNEL, END, MESSAGE)},
    {"assistant_restart_ids": (START, STOP), "prompt_suffix_ids": (START, STOP)},
    {"forbidden_control_ids": (TOOL, TOOL)},
    *({"forbidden_control_ids": (token,)} for token in
      (START, CHANNEL, MESSAGE, END, STOP, 11, 12, 13, 21, 22)),
])
def test_contract_rejects_inconsistent_or_ambiguous_shapes(changes):
    with pytest.raises(ValueError):
        replace(CONTRACT, **changes)


def test_ordinary_id_repetition_and_minimum_shapes_are_legal():
    minimal = replace(CONTRACT, prompt_suffix_ids=(START, 21),
                      assistant_restart_ids=(START, 21),
                      initial_analysis_header_ids=(CHANNEL, 11, MESSAGE),
                      initial_final_header_ids=(CHANNEL, 12, MESSAGE))
    assert HarmonyBoundaryTracker(minimal, special_token_ids=SPECIALS)
    repeated = replace(CONTRACT, prompt_suffix_ids=(START, 21, 21),
                       assistant_restart_ids=(START, 21, 21))
    assert HarmonyBoundaryTracker(repeated, special_token_ids=SPECIALS)


@pytest.mark.parametrize("value", BAD_SEQUENCES + ((START, START), ()))
def test_tracker_rejects_invalid_special_vocabulary(value):
    with pytest.raises(ValueError):
        HarmonyBoundaryTracker(CONTRACT, special_token_ids=value)


@pytest.mark.parametrize("missing", (START, CHANNEL, MESSAGE, END, STOP, TOOL, SCHEMA))
def test_tracker_requires_all_contract_controls(missing):
    with pytest.raises(ValueError):
        HarmonyBoundaryTracker(CONTRACT, special_token_ids=tuple(
            token for token in SPECIALS if token != missing))


@pytest.mark.parametrize("ordinary", (11, 12, 13, 21, 22))
def test_header_and_role_words_cannot_be_declared_special(ordinary):
    with pytest.raises(ValueError):
        HarmonyBoundaryTracker(CONTRACT, special_token_ids=SPECIALS + (ordinary,))


@pytest.mark.parametrize("contract", (None, {}, RESTART, "contract"))
def test_tracker_requires_typed_contract(contract):
    with pytest.raises(TypeError):
        HarmonyBoundaryTracker(contract, special_token_ids=SPECIALS)


def test_public_state_is_read_only_and_exports_are_immutable():
    specials = list(SPECIALS)
    item = HarmonyBoundaryTracker(CONTRACT, special_token_ids=specials)
    specials.clear()
    assert item.contract is CONTRACT
    assert item.special_token_ids == SPECIALS
    assert state(item) == ("header", None, None, (), (), False, (CHANNEL,))
    for name in ("phase", "final_content_start", "final_content_end", "analysis_segments",
                 "observed_token_ids", "contract", "special_token_ids", "is_terminated"):
        with pytest.raises(AttributeError):
            setattr(item, name, None)
    assert not hasattr(item, "__dict__")
    item.observe(REPEATED)
    assert type(item.analysis_segments) is tuple
    assert all(type(segment) is tuple for segment in item.analysis_segments)
    assert type(item.observed_token_ids) is tuple
    assert type(item.allowed_next_ids()) is tuple
    with pytest.raises(TypeError):
        item.allowed_next_ids(SPECIALS)


def test_exact_hand_offsets_for_whole_returned_sequence():
    item = tracker()
    item.observe(REPEATED)
    assert state(item) == ("terminated", 26, 29, ((5, 9), (18, 18)), REPEATED, True, ())
    assert item.observed_token_ids[26:29] == (32, 33, 34)
    assert item.observed_token_ids[5:9] == (11, 13, 30, 31)
    before = state(item)
    item.observe(REPEATED)
    assert state(item) == before


def test_direct_final_and_empty_final_do_not_require_analysis_or_parse_json():
    item = tracker()
    item.observe(DIRECT)
    assert state(item) == ("terminated", 4, 7, (), DIRECT, True, ())
    empty = tracker()
    empty.observe(FINAL + (STOP,))
    assert empty.final_content_start == empty.final_content_end == 4
    assert empty.analysis_segments == ()
    # Final bytes are deliberately not validated here; C2 owns the matcher.
    unparsed = tracker()
    unparsed.observe(FINAL + (0, 999999, STOP))
    assert unparsed.is_terminated


def test_prefix_trie_shared_branch_and_sorted_unique_next_ids():
    item = tracker()
    for prefix, allowed in (
        ((), (CHANNEL,)), ((CHANNEL,), (11,)), ((CHANNEL, 11), (12, 13)),
        ((CHANNEL, 11, 12), (11,)), ((CHANNEL, 11, 12, 11), (MESSAGE,)),
    ):
        item.observe(prefix)
        assert item.phase == "header"
        assert item.allowed_next_ids() == allowed
    item.observe(ANALYSIS)
    assert item.phase == "analysis"
    assert item.allowed_next_ids() is None
    assert item.analysis_segments == ()


def test_restart_phase_includes_exact_role_and_following_header():
    item = tracker()
    ended = ANALYSIS + (END,)
    item.observe(ended)
    assert item.analysis_segments == ((5, 5),)
    for count in range(len(RESTART) + len(FINAL)):
        item.observe(ended + (RESTART + FINAL)[:count])
        assert item.phase == "restart"
        expected = (RESTART[count],) if count < len(RESTART) else {
            0: (CHANNEL,), 1: (11,), 2: (12, 13), 3: (MESSAGE,),
        }[count - len(RESTART)]
        assert item.allowed_next_ids() == expected
    item.observe(ended + RESTART + FINAL)
    assert item.phase == "final"
    assert item.final_content_start == 13
    assert item.final_content_end is None
    assert item.allowed_next_ids() is None


@pytest.mark.parametrize("prefix,phase,segments,start", [
    ((), "header", (), None), ((CHANNEL, 11), "header", (), None),
    (ANALYSIS + (30, 31), "analysis", (), None),
    (ANALYSIS + (30, END), "restart", ((5, 6),), None),
    (ANALYSIS + (END,) + RESTART + ANALYSIS + (30,), "analysis", ((5, 5),), None),
    (FINAL + (32,), "final", (), 4),
])
def test_budget_exhaustion_does_not_silently_close_open_segments(prefix, phase, segments, start):
    item = tracker()
    item.observe(prefix)
    assert (item.phase, item.analysis_segments, item.final_content_start) == (phase, segments, start)
    assert item.final_content_end is None
    assert not item.is_terminated
    before = state(item)
    item.observe(prefix)
    assert state(item) == before


@pytest.mark.parametrize("split", range(len(REPEATED) + 1))
def test_every_two_chunk_split_matches_tokenwise_and_whole_offsets(split):
    whole, tokenwise, chunked = tracker(), tracker(), tracker()
    whole.observe(REPEATED)
    for end in range(len(REPEATED) + 1):
        tokenwise.observe(REPEATED[:end])
        tokenwise.observe(REPEATED[:end])
    chunked.observe(REPEATED[:split])
    chunked.observe(REPEATED)
    assert state(whole) == state(tokenwise) == state(chunked)


def test_all_chunk_partitions_of_direct_final_and_three_chunk_repeated_analysis():
    whole = tracker()
    whole.observe(DIRECT)
    # All combinations of intermediate chunk cuts, not just single-token feeds.
    for cuts in range(1 << (len(DIRECT) - 1)):
        item = tracker()
        for end in range(1, len(DIRECT)):
            if cuts & (1 << (end - 1)):
                item.observe(DIRECT[:end])
        item.observe(DIRECT)
        assert state(item) == state(whole)
    repeated = tracker()
    repeated.observe(REPEATED)
    for first in range(len(REPEATED) + 1):
        for second in range(first, len(REPEATED) + 1):
            item = tracker()
            item.observe(REPEATED[:first])
            item.observe(REPEATED[:second])
            item.observe(REPEATED)
            assert state(item) == state(repeated)


@pytest.mark.parametrize("prefix", ((), (CHANNEL,), ANALYSIS, ANALYSIS + (END,),
                                      ANALYSIS + (END,) + RESTART, FINAL, DIRECT))
@pytest.mark.parametrize("control", (START, CHANNEL, MESSAGE, END, STOP, TOOL, SCHEMA, UNKNOWN))
def test_special_controls_only_allowed_at_their_declared_boundary(prefix, control):
    item = tracker()
    item.observe(prefix)
    legal = item.allowed_next_ids()
    if legal is None:
        allowed = control == (END if item.phase == "analysis" else STOP)
    else:
        allowed = control in legal
    if allowed:
        item.observe(prefix + (control,))
    else:
        before = state(item)
        with pytest.raises(UnsupportedHarmonyChannel):
            item.observe(prefix + (control,))
        assert state(item) == before


@pytest.mark.parametrize("tokens", [
    (21,), (CHANNEL, 99), (CHANNEL, 11, 99),  # recipient/commentary-like headers
    ANALYSIS + (END,) + (21,),
    ANALYSIS + (END,) + RESTART + (CHANNEL, 99),
    ANALYSIS + (END,) + RESTART + (CHANNEL, 11, 99),
    DIRECT + (32,), DIRECT + (STOP,),
    FINAL + (32, END) + RESTART + FINAL,  # no second final body
])
def test_malformed_transitions_fail_with_local_protocol_error(tokens):
    assert issubclass(UnsupportedHarmonyChannel, ValueError)
    with pytest.raises(UnsupportedHarmonyChannel):
        tracker().observe(tokens)


@pytest.mark.parametrize("replacement", ((), (CHANNEL,), (CHANNEL, 12), (CHANNEL, 11, 13)))
def test_prefix_cannot_be_shortened_or_rewritten(replacement):
    item = tracker()
    item.observe(ANALYSIS[:3])
    before = state(item)
    with pytest.raises(UnsupportedHarmonyChannel):
        item.observe(replacement)
    assert state(item) == before


@pytest.mark.parametrize("tokens", BAD_SEQUENCES)
def test_bad_observation_objects_are_rejected_without_state_changes(tokens):
    item = tracker()
    item.observe(ANALYSIS)
    before = state(item)
    with pytest.raises(ValueError):
        item.observe(tokens)
    assert state(item) == before


def test_input_copy_and_invalid_append_preserve_only_the_accepted_prefix():
    item = tracker()
    supplied = list(ANALYSIS)
    item.observe(supplied)
    supplied.clear()
    assert item.observed_token_ids == ANALYSIS
    with pytest.raises(UnsupportedHarmonyChannel):
        item.observe(ANALYSIS + (30, 31, UNKNOWN, 32))
    assert item.observed_token_ids == ANALYSIS + (30, 31)
    assert item.analysis_segments == ()
    item.observe(iter(ANALYSIS + (30, 31)))
    assert item.phase == "analysis"


def test_module_can_load_in_isolation_with_only_standard_library_imports():
    # The existing package __init__ exports other inference components. Load this
    # file independently, in a fresh isolated process blocking any ML imports.
    path = Path(__file__).parents[1] / "llm_bias/core/inference/harmony_channels.py"
    script = """
import builtins
import importlib.util
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'xgrammar'}:
        raise AssertionError('unexpected ML dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
spec = importlib.util.spec_from_file_location('_isolated_harmony_channels', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
contract = module.HarmonyTokenContract(
    (100, 1), (101, 2, 102), (101, 3, 102), 103, (100, 1), 104, ())
tracker = module.HarmonyBoundaryTracker(contract, special_token_ids=(100, 101, 102, 103, 104))
tracker.observe((101, 3, 102, 4, 104))
assert tracker.is_terminated
assert (tracker.final_content_start, tracker.final_content_end) == (3, 4)
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(path)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
