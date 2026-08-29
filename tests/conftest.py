"""Shared pytest fixtures for the regression suite."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _no_real_cuda(monkeypatch) -> None:
    """Keep unit tests independent of a real GPU on the host.

    Pipeline helpers call ``torch.use_deterministic_algorithms(True)`` when
    ``torch.cuda.is_available()``; on a GPU host that leaks the global
    deterministic flag into later tests.  Tests that exercise the
    CUDA-available code path monkeypatch ``torch.cuda.is_available``
    themselves, which overrides this guard.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
