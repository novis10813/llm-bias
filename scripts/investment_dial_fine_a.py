"""A-only V1 auxiliary diagnostic; never refits or replaces calibration."""
import argparse
import json

import torch

from llm_bias.core.artifact_paths import file_sha256
from llm_bias.investment_dial import pipeline as p
from llm_bias.investment_dial.analysis import summary

DELTAS = tuple(i / 4 for i in range(9))
COORDINATE = (15, 8490)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source-run', required=True, help='Complete original V1 calibration run directory')
    result.add_argument('--model', required=True, help='Identical local checkpoint directory')
    result.add_argument('--run-id', required=True, help='New diagnostic run ID; never reuse a V1 ID')
    result.add_argument('--artifact-root', default='artifacts')
    result.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    return result


def select_rows(rows, result):
    selected = result.get('selected')
    if selected is None or (selected['layer'], selected['neuron']) != COORDINATE:
        raise ValueError('Expected original selected coordinate (15, 8490)')
    selected_rows = [r for r in rows if r['split'] == 'A' and r['positive_count'] == 2]
    if len(selected_rows) != 340 or len({r['ticker'] for r in selected_rows}) != 85:
        raise ValueError('Expected 340 A-only balanced prompts from 85 companies')
    return selected_rows


def run(args):
    if args.device == 'cpu' and torch.cuda.is_available():
        raise ValueError("CPU execution requires CUDA_VISIBLE_DEVICES='' before launch")
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    bundle = p.verified_run(args.source_run, 'investment-dial-calibration', p.REQUIRED)
    # Fail before loading weights if the selected coordinate or A subset is wrong.
    select_rows(bundle[0]['prepare/trials.json'], bundle[0]['analyze/result.json'])
    model, tokenizer, device = p.load_model(args.model, dtype=torch.bfloat16)
    parent, rows, result, digest = p._parent(bundle, args.model, tokenizer, model, device)
    rows = select_rows(rows, result)
    protocol = dict(
        schema_version=1, protocol_version='investment-dial-fine-a-v1',
        purpose='A-only auxiliary curve diagnostic; no refit; no B/test generation',
        parent_run=str(args.source_run), parent_manifest_sha256=digest,
        model_identity=parent['model_identity'], runtime=p._runtime(model, tokenizer, device),
        parent_runtime=parent['runtime'], source_identity=p._source(),
        operator_sha256=file_sha256(__file__), layer=15, neuron=8490, deltas=list(DELTAS),
        max_new_tokens=256, note='Delta zero remeasured; hardware changes may alter numerical results')
    with p.run_context(args.model, 'investment-dial-fine-a', args.run_id,
                       artifact_root=args.artifact_root) as output, p.frozen_eval(model):
        print('RUN', output.run_directory, flush=True)
        with output.stage('prepare') as stage:
            p.write(output, 'prepare/protocol.json', protocol)
            p.write(output, 'prepare/trials.json', rows)
            stage.count(len(rows))
        stats = []
        with output.stage('forward') as stage:
            for i, delta in enumerate(DELTAS):
                records = []
                for j, row in enumerate(rows):
                    records.extend(p._decisions(model, tokenizer, device, [row], COORDINATE, delta, 256))
                    if (j + 1) % 20 == 0:
                        print(f'delta={delta} trials={j+1}/{len(rows)}', flush=True)
                p.write(output, f'forward/delta-{i}.jsonl', records)
                stats.append(dict(delta=delta, **summary(records)))
                print(json.dumps(stats[-1]), flush=True)
            stage.count(len(rows) * len(DELTAS))
        with output.stage('analyze') as stage:
            p.write(output, 'analyze/result.json', dict(
                protocol_sha256=p.object_sha256(protocol), certified=False, summaries=stats))
            stage.count(len(stats))
        output.finalize(required_stages={'prepare', 'forward', 'analyze'})
        return output.run_directory


if __name__ == '__main__':
    run(parser().parse_args())
