"""Strict, artifact-only readers for synthetic entity-bias visualization."""
from .contract import (
    BASELINE_FIELDS,
    ENTITY_POOL_FIELDS,
    LOCALIZATION_FIELDS,
    RESULT_FIELDS,
    REQUIRED_OUTPUTS,
)

RAW_FIELDS = RESULT_FIELDS
REQUIRED_ARTIFACTS = REQUIRED_OUTPUTS
from .reader import ArtifactBundle, ArtifactContractError, ValidatedRun, read_run, validate_run

# Keep the reader importable without plotting/runtime dependencies. Visualization
# helpers remain available through lazy attribute resolution for compatibility.
_LAZY = {
    "summarize_all": (".summaries", "summarize_all"),
    "summarize_template": (".summaries", "summarize_template"),
    "summarize_tier": (".summaries", "summarize_tier"),
    "summarize_sector": (".summaries", "summarize_sector"),
    "summarize_ticker": (".summaries", "summarize_ticker"),
    "summarize_localization": (".summaries", "summarize_localization"),
    "make_plots": (".plots", "make_plots"),
    "render_dashboard": (".dashboard", "render_dashboard"),
    "write_dashboard": (".dashboard", "write_dashboard"),
    "visualize_run": (".runner", "visualize_run"),
}

def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        value = getattr(import_module(_LAZY[name][0], __name__), _LAZY[name][1])
        globals()[name] = value
        return value
    raise AttributeError(name)

__all__ = [
    "ArtifactBundle", "ArtifactContractError", "ValidatedRun", "read_run", "validate_run",
    "BASELINE_FIELDS", "ENTITY_POOL_FIELDS", "LOCALIZATION_FIELDS", "RAW_FIELDS",
    "REQUIRED_ARTIFACTS", *_LAZY,
]
