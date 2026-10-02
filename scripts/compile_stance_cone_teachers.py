"""Compile immutable full-fit cone teachers on CPU, never checkpoint weights."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.stance_baseline_parent import load_completed_baseline
from llm_bias.core.stance_baseline_merged import load_merged_baseline
from llm_bias.core.stance_cone_teachers import compile_cone_teachers, write_teacher_pack


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'parent', 'model', 'output-dir'):
        result.add_argument('--' + name, required=True, type=Path)
    result.add_argument('--recovery', type=Path)
    return result


def verify_checkpoint_metadata(checkpoint, expected):
    """Verify the same local metadata selection as the accepted baseline runner."""
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_dir() or str(checkpoint) != expected['resolved_path']:
        raise ValueError('require exact parent local checkpoint path')
    files = {p.name: sha256_bytes(p.read_bytes()) for p in checkpoint.iterdir()
             if p.is_file() and (p.name in ('config.json', 'generation_config.json',
                'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
                'chat_template.jinja') or p.name.startswith('tokenizer'))}
    if files != expected['metadata_file_sha256']:
        raise ValueError('checkpoint tokenizer/config metadata differs from full parent')
    return checkpoint


def run(args):
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError('teacher pack requires a fresh output directory')
    inputs = load_baseline_inputs(args.inputs)
    parent = (load_merged_baseline(args.parent, args.recovery, inputs=inputs)
              if args.recovery is not None else load_completed_baseline(args.parent, inputs=inputs))
    # A named unsupported report needs no speculative channel tokenizer policy.
    tokenizer = None
    if parent.metadata['bindings']['generation_policy']['policy']['channel_policy'] == 'plain_json':
        checkpoint = verify_checkpoint_metadata(args.model, parent.metadata['bindings']['model'])
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                                  trust_remote_code=False, use_fast=True)
    pack = compile_cone_teachers(inputs, parent, tokenizer)
    files = [Path(__file__), ROOT / 'uv.lock', *sorted((ROOT / 'llm_bias/core').rglob('*.py'))]
    config = dict(kind='stance_cone_teacher_compiler_config_v1',
        inputs_manifest_sha256=inputs.manifest_sha256, parent_sha256=parent.parent_sha256,
        model=parent.metadata['bindings']['model'],
        execution='CPU_tokenizer_and_grammar_only_no_checkpoint_weights',
        backend=dict(python=platform.python_version(), transformers=version('transformers'),
                     tokenizers=version('tokenizers'), xgrammar=version('xgrammar')),
        code=dict(git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                   text=True).strip(),
                  source_sha256={str(p.relative_to(ROOT)): sha256_bytes(p.read_bytes()) for p in files}))
    write_teacher_pack(args.output_dir, pack, config)
    print(canonical_json_bytes(pack['manifest']).decode())
    return 0 if pack['manifest']['accepted'] else 2


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
