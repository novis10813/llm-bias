"""One-row CUDA diagnostic of the approved full pack; never a new cohort."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from importlib.metadata import version
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from types import SimpleNamespace

# Also support `uv run python scripts/smoke_stance_checkpoint.py` from the repo.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.model import load_model
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.prompt_input.decision_prompt import WrapperPolicy, render_decision_prompt
from llm_bias.core.inference.structured_output import (
    StructuredGenerationPolicy, compile_decision_grammar, _declared_tokens, _fast_backend,
)
from llm_bias.core.inference.harmony_channels import HarmonyTokenContract
from llm_bias.core.inference.harmony_generation import compile_harmony_decision_grammar
from llm_bias.core.inference.stance_noop_execution import execute_prompt_noop


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', required=True, type=Path)
    result.add_argument('--inputs', required=True, type=Path)
    result.add_argument('--output', required=True, type=Path)
    result.add_argument('--stop-token-id', required=True, action='append', type=int)
    result.add_argument('--channel-policy', choices=('plain_json', 'harmony_no_tools'), default='plain_json')
    result.add_argument('--dtype', choices=('bfloat16', 'native'), default='bfloat16')
    result.add_argument('--max-new-tokens', type=int, default=512)
    result.add_argument('--timeout-seconds', type=float, default=180)
    result.add_argument('--layer', type=int, default=0)
    result.add_argument('--hook-site', choices=('pre', 'mid', 'post'), default='pre')
    return result


def select_diagnostic(inputs):
    # Selection is deliberately downstream of the strict, pinned full-pack loader.
    if (len(inputs.members), len(inputs.issuer_by_ticker.values()), len(inputs.pairs)) != (503, 503, 2012):
        raise ValueError('require the full 503-member / 2012-pair pack')
    if len(set(inputs.issuer_by_ticker.values())) != 500:
        raise ValueError('require 500 issuers')
    pair = min(inputs.pairs, key=lambda p: (p.ticker, p.condition, p.trial_id))
    member = next(m for m in inputs.members if m.ticker == pair.ticker)
    return member, pair


def generation_adapter(model):
    hf_model = getattr(model, 'hf_model', None)
    if hf_model is None:
        hf_model = getattr(model, '_hf_model', None)
    if hf_model is None:
        raise ValueError('loaded wrapper has no underlying HF model')
    return SimpleNamespace(hf_model=hf_model, layers=model.layers, tokenizer=model.tokenizer)


def compile_smoke_grammar(tokenizer, head_vocab_size, stop_ids, channel_policy):
    if channel_policy == 'plain_json':
        return compile_decision_grammar(tokenizer, head_vocab_size, stop_ids)
    _, specials = _declared_tokens(tokenizer, _fast_backend(tokenizer), head_vocab_size)
    controls = {}
    for literal in ('<|start|>', '<|channel|>', '<|message|>', '<|end|>', '<|return|>'):
        token_id = tokenizer.convert_tokens_to_ids(literal)
        if token_id not in specials or tokenizer.encode(literal, add_special_tokens=False) != [token_id]:
            raise ValueError(f'Harmony control is not an exact declared special: {literal}')
        controls[literal] = token_id
    if list(stop_ids) != [controls['<|return|>']]:
        raise ValueError('Harmony stop IDs must be exactly the singleton <|return|> ID')
    encode = lambda text: tuple(tokenizer.encode(text, add_special_tokens=False))
    restart = encode('<|start|>assistant')
    analysis = encode('<|channel|>analysis<|message|>')
    final = encode('<|channel|>final<|message|>')
    admitted = set(restart + analysis + final) | set(controls.values())
    contract = HarmonyTokenContract(
        restart, analysis, final, controls['<|end|>'], restart, controls['<|return|>'],
        tuple(sorted(specials - admitted)))
    return compile_harmony_decision_grammar(tokenizer, head_vocab_size, contract)


def run_smoke(args, record):
    record['phase'] = 'inputs'
    inputs = load_baseline_inputs(args.inputs)
    member, pair = select_diagnostic(inputs)
    record.update(full_manifest_hash=inputs.manifest_sha256,
                  selected_key=[pair.ticker, pair.condition, pair.trial_id])
    record['phase'] = 'metadata'
    checkpoint = args.model.resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError('model must be a local checkpoint directory')
    files = {p.name: sha256_bytes(p.read_bytes()) for p in checkpoint.iterdir()
             if p.is_file() and (p.name == 'config.json' or
                                 p.name == 'generation_config.json' or
                                 p.name.startswith('tokenizer') or
                                 p.name in ('special_tokens_map.json', 'added_tokens.json',
                                            'vocab.json', 'merges.txt', 'chat_template.jinja'))}
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if not re.fullmatch('[0-9a-f]{40}', head):
        raise ValueError('expected 40-hex git HEAD provenance')
    record.update(model_path=str(checkpoint), checkpoint_file_sha256=files,
                  provenance={'git_head': head, 'git_head_kind': 'git_object_id'},
                  versions={'python': platform.python_version(), 'torch': torch.__version__,
                            'transformers': version('transformers'), 'xgrammar': version('xgrammar'),
                            'cuda': torch.version.cuda},
                  environment={k: v for k, v in os.environ.items()
                               if k.startswith('LAB') or k == 'CUDA_VISIBLE_DEVICES'},
                  cuda={'available': torch.cuda.is_available(),
                        'logical_visible_device_count': torch.cuda.device_count()})
    record['phase'] = 'load_model'
    if not torch.cuda.is_available():
        raise RuntimeError('this diagnostic requires CUDA; CPU fallback is forbidden')
    model, returned_tokenizer, device = load_model(str(checkpoint), device_map=None, dtype='native' if args.dtype == 'native' else torch.bfloat16)
    if torch.device(device).type != 'cuda':
        raise RuntimeError('loader returned a non-CUDA device')
    # jlens exposes _hf_model; generation contracts use a public hf_model.
    # Adapt references locally without modifying the shared loader or checkpoint.
    model = generation_adapter(model)
    model.hf_model.eval()
    # jlens force_bos may mutate/attach the tokenizer: use the actual attachment.
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        tokenizer = getattr(model.hf_model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError('loaded model has no attached tokenizer')
    del returned_tokenizer
    weight = model.hf_model.get_output_embeddings().weight
    if weight.device.type != 'cuda':
        raise RuntimeError('actual output head is not on CUDA')
    config = model.hf_model.config
    text_config = config.get_text_config() if callable(getattr(config, 'get_text_config', None)) else config
    width = getattr(text_config, 'hidden_size', None)
    if width is None:
        width = weight.shape[1]
    record['phase'] = 'compile_and_render'
    capability = compile_smoke_grammar(tokenizer, weight.shape[0], args.stop_token_id, args.channel_policy)
    wrapper = WrapperPolicy(use_chat_template=True, add_special_tokens=False,
                            enable_thinking=args.channel_policy == 'harmony_no_tools')
    prompt = render_decision_prompt(tokenizer, member, pair, wrapper_policy=wrapper)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = args.stop_token_id[0]
    policy = StructuredGenerationPolicy(max_new_tokens=args.max_new_tokens, use_cache=True,
                                       pad_token_id=pad, timeout_seconds=args.timeout_seconds,
                                       channel_policy=args.channel_policy)
    embedding = model.hf_model.get_input_embeddings().weight
    if not embedding.dtype.is_floating_point or not weight.dtype.is_floating_point:
        raise ValueError('smoke requires floating embedding and head weights')
    vector_dtype = embedding.dtype
    operands = dict(layer=args.layer, hook_site=args.hook_site, scope='prompt_only',
                    hidden_width=width, direction='ones', zero_dose=0.0, replacement_dose=1,
                    head_vocab_size=weight.shape[0], stop_token_ids=args.stop_token_id,
                    dtype=args.dtype, vector_dtype=str(vector_dtype), head_weight_dtype=str(weight.dtype),
                    embedding_weight_dtype=str(embedding.dtype), device=str(device), device_map=None,
                    prompt_sha256=prompt.prompt_sha256, wrapper_policy=prompt.wrapper_policy,
                    policy=asdict(policy), model_path=str(checkpoint),
                    checkpoint_file_sha256=files, provenance=record['provenance'])
    record.update(operands=operands, config_hash=sha256_json(operands))
    prompt_ids = torch.tensor([prompt.inference_token_ids], dtype=torch.long, device=device)
    vector = torch.ones(width, dtype=vector_dtype, device=device)
    record['phase'] = 'execute_prompt_noop'
    execution = execute_prompt_noop(model, tokenizer, prompt_ids, capability, policy=policy,
                                   layer=args.layer, hook_site=args.hook_site, zero_vector=vector,
                                   config_hash=record['config_hash'])
    record.update(arms={name: None if getattr(execution, name) is None else getattr(execution, name).to_dict()
                        for name in ('baseline', 'repeat', 'zero', 'self_replacement')},
                  gate=None if execution.gate is None else execution.gate.to_dict(),
                  diagnostics={'zero': execution.zero_diagnostics, 'self_replacement': execution.self_diagnostics},
                  halt_reason=execution.halt_reason, completed=execution.completed, passed=execution.passed,
                  phase='completed')
    return 0 if execution.passed else 1


def main(argv=None):
    args = parser().parse_args(argv)
    # Reserve the fresh output before any model work; never truncate an old run.
    with args.output.open('xb') as output:
        record = dict(kind='diagnostic_smoke', research_eligible=False, new_cohort=False,
                      planned_rows=2012, diagnostic_rows=1, completed=False, passed=False,
                      phase='start', halt_reason=None)
        try:
            with redirect_stdout(sys.stderr):
                code = run_smoke(args, record)
        except Exception as exc:
            message = str(exc).encode('utf-8', errors='replace').decode('utf-8')
            record.update(error={'type': type(exc).__name__, 'message': message},
                          halt_reason='exception', research_eligible=False, passed=False)
            code = 1
        output.write(canonical_json_bytes(record) + b'\n')
        output.flush()
        os.fsync(output.fileno())
    return code


if __name__ == '__main__':
    raise SystemExit(main())
