"""Interactive Jacobian-lens readout for an arbitrary prompt."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from jspace_viz.analysis import read_grid
from jspace_viz.lens import JacobianLens
from jspace_viz.model import WrappedModel

from llm_bias.core.lens_artifacts import (
    canonical_lens_path,
    validate_lens_for_model,
)
from llm_bias.core.model import load_model as load_lens_model
from llm_bias.core.prompting import format_prompt

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@dataclass(frozen=True)
class PromptReadoutState:
    """Model and fitted lens kept resident for interactive requests."""

    model: WrappedModel
    lens: JacobianLens
    model_id: str
    lens_path: str


class PromptReadoutRequest(BaseModel):
    """Validated controls for one prompt readout."""

    prompt: str = Field(min_length=1, max_length=20_000)
    mode: Literal["jlens", "logit"] = "jlens"
    top_k: int = Field(default=8, ge=1, le=32)
    max_seq_len: int = Field(default=256, ge=8, le=2048)
    chat: bool = True
    enable_thinking: bool = False
    generate_continuation: bool = False
    max_new_tokens: int = Field(default=64, ge=1, le=256)


@dataclass(frozen=True)
class _LogitLayerSelection:
    """Minimal read-grid lens interface selecting every decoder layer."""

    source_layers: list[int]


@torch.no_grad()
def _generate_response(
    model: WrappedModel,
    formatted_prompt: str,
    *,
    max_seq_len: int,
    max_new_tokens: int,
) -> tuple[str, int]:
    """Generate one deterministic response from the exact readout prompt."""
    input_ids = model.encode(formatted_prompt, max_length=max_seq_len)
    generated = model.hf_model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=model.tokenizer.eos_token_id,
    )
    response_ids = generated[0, input_ids.shape[1] :]
    response = model.tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return response, int(response_ids.numel())


def _model_info(state: PromptReadoutState) -> dict[str, Any]:
    fitted_layers = list(state.lens.source_layers)
    expected_source_layers = list(range(state.model.n_layers - 1))
    chat_template = getattr(state.model.tokenizer, "chat_template", None)
    canonical_path = canonical_lens_path(state.model_id)
    resolved_lens_path = Path(state.lens_path).resolve()
    return {
        "model_id": state.model_id,
        "device": str(state.model.device),
        "n_layers": state.model.n_layers,
        "d_model": state.model.d_model,
        "fitted_layers": fitted_layers,
        "missing_fitted_layers": sorted(set(expected_source_layers) - set(fitted_layers)),
        "fitted_layer_count": len(fitted_layers),
        "expected_fitted_layer_count": len(expected_source_layers),
        "lens_n_prompts": state.lens.n_prompts,
        "lens_source": state.lens_path,
        "lens_is_model_canonical": resolved_lens_path == canonical_path.resolve(),
        "canonical_lens_source": str(canonical_path),
        "has_chat_template": bool(chat_template),
        "supports_enable_thinking": (
            isinstance(chat_template, str)
            and "enable_thinking" in chat_template
        ),
        "max_top_k": 32,
        "default_max_seq_len": 256,
    }


def read_prompt(
    state: PromptReadoutState,
    request: PromptReadoutRequest,
) -> dict[str, Any]:
    """Compute a compact layer-by-position readout without saving activations."""
    prompt = request.prompt.strip()
    if not prompt:
        raise ValueError("prompt must contain non-whitespace text")
    if request.chat and not getattr(state.model.tokenizer, "chat_template", None):
        raise ValueError(
            "chat formatting is unavailable for this tokenizer; disable Chat template"
        )
    formatted_prompt = format_prompt(
        state.model.tokenizer,
        prompt,
        use_chat_template=request.chat,
        enable_thinking=request.enable_thinking,
    )

    readout_lens = (
        state.lens
        if request.mode == "jlens"
        else _LogitLayerSelection(list(range(state.model.n_layers - 1)))
    )
    result = read_grid(
        state.model,
        readout_lens,
        formatted_prompt,
        mode=request.mode,
        top_k=request.top_k,
        max_seq_len=request.max_seq_len,
        chat=False,
        generate_continuation=False,
    )
    if request.generate_continuation:
        response, response_token_count = _generate_response(
            state.model,
            result["prompt"],
            max_seq_len=request.max_seq_len,
            max_new_tokens=request.max_new_tokens,
        )
        result["continuation"] = response
        result["response_token_count"] = response_token_count
    else:
        result["continuation"] = None
        result["response_token_count"] = 0
    result["input_prompt"] = prompt
    result["chat_enabled"] = request.chat
    result["thinking_enabled"] = (
        request.chat and request.enable_thinking
    )
    result["requested_max_seq_len"] = request.max_seq_len
    result["truncated"] = (
        result["prompt_len"] >= request.max_seq_len
        and len(
            state.model.tokenizer(
                result["prompt"],
                add_special_tokens=True,
                truncation=False,
            ).input_ids
        )
        > request.max_seq_len
    )
    return result


def create_app(state: PromptReadoutState) -> FastAPI:
    """Create the HTTP app around an already-loaded state (test-friendly)."""
    app = FastAPI(title="Prompt Lens Explorer")

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return _model_info(state)

    @app.post("/api/readout")
    def prompt_readout(request: PromptReadoutRequest) -> dict[str, Any]:
        try:
            return read_prompt(state, request)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("prompt readout failed")
            raise HTTPException(500, str(exc)) from exc

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "prompt_readout.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def build_app(model_name: str, lens_path: str | None = None) -> FastAPI:
    """Load the model and its model-scoped canonical lens once.

    An explicit ``lens_path`` remains an opt-in experimental override; the
    default path is always resolved through the model-specific lens authority.
    """
    resolved_lens_path = str(
        Path(lens_path) if lens_path is not None else canonical_lens_path(model_name)
    )
    lens_model, tokenizer, _device = load_lens_model(model_name)
    model = WrappedModel(lens_model._hf_model, tokenizer)
    lens = JacobianLens.load(resolved_lens_path)
    validate_lens_for_model(
        model=model,
        lens=lens,
        model_name=model_name,
        lens_path=resolved_lens_path,
        require_complete=True,
    )
    return create_app(
        PromptReadoutState(model, lens, model_name, resolved_lens_path)
    )
