"""Pure token boundaries for a pinned, no-tool Harmony assistant turn.

This module knows neither token spellings nor JSON legality. Callers supply a
verified contract and special vocabulary; schema-aware generation owns final
body validation and vocabulary/head bounds.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


class UnsupportedHarmonyChannel(ValueError):
    """The continuation violates the explicitly bound no-tool protocol."""


def _token_id(value: int, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a genuine nonbool nonnegative Python integer")
    return value


def _token_tuple(value: tuple[int, ...], label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{label} must be an iterable of Python integer IDs")
    try:
        tokens = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable of Python integer IDs") from exc
    for token in tokens:
        _token_id(token, label)
    return tokens


@dataclass(frozen=True, slots=True)
class HarmonyTokenContract:
    """Immutable explicit tokens, with no checkpoint-specific default bindings."""

    prompt_suffix_ids: tuple[int, ...]
    initial_analysis_header_ids: tuple[int, ...]
    initial_final_header_ids: tuple[int, ...]
    message_end_id: int
    assistant_restart_ids: tuple[int, ...]
    final_stop_id: int
    forbidden_control_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("prompt_suffix_ids", "initial_analysis_header_ids",
                     "initial_final_header_ids", "assistant_restart_ids",
                     "forbidden_control_ids"):
            object.__setattr__(self, name, _token_tuple(getattr(self, name), name))
        _token_id(self.message_end_id, "message_end_id")
        _token_id(self.final_stop_id, "final_stop_id")
        analysis = self.initial_analysis_header_ids
        final = self.initial_final_header_ids
        restart = self.assistant_restart_ids
        if len(analysis) < 3 or len(final) < 3:
            raise ValueError("headers need channel marker, ordinary name IDs and message marker")
        if len(restart) < 2:
            raise ValueError("assistant_restart_ids needs start marker and ordinary role IDs")
        if self.prompt_suffix_ids != restart:
            raise ValueError("prompt_suffix_ids must equal assistant_restart_ids")
        if analysis[0] != final[0] or analysis[-1] != final[-1]:
            raise ValueError("headers must share their channel and message markers")
        if analysis[:len(final)] == final or final[:len(analysis)] == analysis:
            raise ValueError("header alternatives must not equal or prefix each other")
        controls = (self.message_end_id, self.final_stop_id, analysis[0], analysis[-1], restart[0])
        if len(set(controls)) != len(controls):
            raise ValueError("message-end, final-stop and header/restart markers must be distinct")
        words = analysis[1:-1] + final[1:-1] + restart[1:]
        if set(words).intersection(controls):
            raise ValueError("role/channel name IDs must not be admitted controls")
        forbidden = self.forbidden_control_ids
        if len(set(forbidden)) != len(forbidden):
            raise ValueError("forbidden_control_ids must be unique")
        admitted = set(analysis + final + restart + controls)
        if admitted.intersection(forbidden):
            raise ValueError("forbidden controls must be disjoint from every admitted ID")


def _control_ids(contract: HarmonyTokenContract) -> tuple[int, ...]:
    return (contract.message_end_id, contract.final_stop_id,
            contract.initial_analysis_header_ids[0], contract.initial_analysis_header_ids[-1],
            contract.assistant_restart_ids[0]) + contract.forbidden_control_ids


@dataclass(slots=True)
class _HeaderNode:
    children: dict[int, _HeaderNode] = field(default_factory=dict)
    channel: Literal["analysis", "final"] | None = None


class HarmonyBoundaryTracker:
    """Incrementally consume entire, append-only continuation prefixes.

    All public state is read-only and exported as immutable values. Completed
    spans use absolute, half-open continuation offsets, excluding end controls.
    On a protocol error, only tokens preceding the offending token have been
    accepted; the offending token never changes state. Invalid input objects or
    rewritten prefixes are rejected before consumption.
    """

    __slots__ = ("_contract", "_special_token_ids", "_special_set", "_phase",
                 "_final_content_start", "_final_content_end", "_analysis_segments",
                 "_analysis_start", "_observed", "_header_root", "_header_node",
                 "_restart_index")

    def __init__(self, contract: HarmonyTokenContract, *, special_token_ids: tuple[int, ...]) -> None:
        if not isinstance(contract, HarmonyTokenContract):
            raise TypeError("contract must be a HarmonyTokenContract")
        specials = _token_tuple(special_token_ids, "special_token_ids")
        special_set = frozenset(specials)
        if len(special_set) != len(specials):
            raise ValueError("special_token_ids must be unique")
        if not special_set.issuperset(_control_ids(contract)):
            raise ValueError("special_token_ids must include every contract control")
        words = (contract.initial_analysis_header_ids[1:-1]
                 + contract.initial_final_header_ids[1:-1] + contract.assistant_restart_ids[1:])
        if special_set.intersection(words):
            raise ValueError("role/channel name IDs must be ordinary, not special")
        self._contract = contract
        self._special_token_ids = specials
        self._special_set = special_set
        self._phase: Literal["header", "analysis", "restart", "final", "terminated"] = "header"
        self._final_content_start: int | None = None
        self._final_content_end: int | None = None
        self._analysis_segments: tuple[tuple[int, int], ...] = ()
        self._analysis_start: int | None = None
        self._observed: list[int] = []
        self._header_root = _HeaderNode()
        for header, channel in ((contract.initial_analysis_header_ids, "analysis"),
                                (contract.initial_final_header_ids, "final")):
            node = self._header_root
            for token in header:
                if token not in node.children:
                    node.children[token] = _HeaderNode()
                node = node.children[token]
            node.channel = channel
        self._header_node = self._header_root
        self._restart_index = 0

    @property
    def contract(self) -> HarmonyTokenContract:
        return self._contract

    @property
    def special_token_ids(self) -> tuple[int, ...]:
        return self._special_token_ids

    @property
    def phase(self) -> Literal["header", "analysis", "restart", "final", "terminated"]:
        return self._phase

    @property
    def final_content_start(self) -> int | None:
        return self._final_content_start

    @property
    def final_content_end(self) -> int | None:
        return self._final_content_end

    @property
    def analysis_segments(self) -> tuple[tuple[int, int], ...]:
        return self._analysis_segments

    @property
    def observed_token_ids(self) -> tuple[int, ...]:
        return tuple(self._observed)

    @property
    def is_terminated(self) -> bool:
        return self._phase == "terminated"

    def allowed_next_ids(self) -> tuple[int, ...] | None:
        """Exact header/restart alternatives, None for bodies, () after return.

        None does not authorize arbitrary special IDs. Analysis permits ordinary
        vocabulary and message-end; the final matcher owns ordinary/stop legality.
        This method reports boundaries only and never masks or synthesizes logits.
        """
        if self._phase == "terminated":
            return ()
        if self._phase in ("analysis", "final"):
            return None
        restart = self._contract.assistant_restart_ids
        if self._phase == "restart" and self._restart_index < len(restart):
            return (restart[self._restart_index],)
        return tuple(sorted(self._header_node.children))

    def observe(self, token_ids: tuple[int, ...]) -> None:
        """Validate the full prefix, then accept newly appended IDs in order."""
        tokens = _token_tuple(token_ids, "token_ids")
        count = len(self._observed)
        if len(tokens) < count or tokens[:count] != tuple(self._observed):
            raise UnsupportedHarmonyChannel("previously observed continuation prefix changed")
        for index in range(count, len(tokens)):
            self._consume(tokens[index], index)
            self._observed.append(tokens[index])

    def _consume(self, token: int, index: int) -> None:
        if self._phase == "terminated":
            raise UnsupportedHarmonyChannel(f"token after final-stop at continuation index {index}")
        if self._phase in ("header", "restart"):
            restart = self._contract.assistant_restart_ids
            if self._phase == "restart" and self._restart_index < len(restart):
                if token != restart[self._restart_index]:
                    raise UnsupportedHarmonyChannel(f"malformed assistant restart at index {index}")
                self._restart_index += 1
                return
            node = self._header_node.children.get(token)
            if node is None:
                raise UnsupportedHarmonyChannel(f"unsupported assistant header at index {index}")
            self._header_node = node
            if node.channel == "analysis":
                self._phase = "analysis"
                self._analysis_start = index + 1
            elif node.channel == "final":
                self._phase = "final"
                self._final_content_start = index + 1
            return
        if self._phase == "analysis":
            if token == self._contract.message_end_id:
                # This start is assigned only by a completed analysis header.
                assert self._analysis_start is not None
                self._analysis_segments += ((self._analysis_start, index),)
                self._analysis_start = None
                self._phase = "restart"
                self._restart_index = 0
                self._header_node = self._header_root
                return
        elif token == self._contract.final_stop_id:
            self._final_content_end = index
            self._phase = "terminated"
            return
        if token in self._special_set:
            raise UnsupportedHarmonyChannel(f"unsupported special token {token} in {self._phase} at index {index}")
