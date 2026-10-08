"""Validate and export a fresh GPT effective baseline, without generation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_bias.core.artifact_paths import canonical_json_bytes
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.stance_baseline_merged import load_merged_baseline, materialize_merged_baseline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'original', 'recovery', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        view = load_merged_baseline(args.original, args.recovery,
                                    inputs=load_baseline_inputs(args.inputs))
        materialize_merged_baseline(view, args.output_dir)
        print(canonical_json_bytes(dict(kind='effective_merged_baseline_v1',
            content_sha256=view.content_sha256, planned=2012, original_rows=1981,
            recovery_rows=31, class_counts=view.summary['class_counts'],
            mixed_generation_policy=True, output_dir=str(args.output_dir))).decode())
        return 0
    except Exception as exc:
        print(canonical_json_bytes(dict(kind='merged_baseline_failure',
            error_type=type(exc).__name__, message=str(exc))).decode(), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
