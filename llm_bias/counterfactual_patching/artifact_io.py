"""Backward-compatible JSONL facade for counterfactual workflows."""
from llm_bias.core.artifacts.io import read_jsonl, write_jsonl, sha256_file

__all__ = ["read_jsonl", "write_jsonl", "sha256_file"]
