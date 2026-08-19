"""Artifact-only statistical analysis for synthetic entity-bias runs."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "analyze_run":
        from .runner import analyze_run

        globals()[name] = analyze_run
        return analyze_run
    raise AttributeError(name)


__all__ = ["analyze_run"]
