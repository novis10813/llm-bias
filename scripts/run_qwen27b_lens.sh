#!/usr/bin/env bash
set -euo pipefail
set -o pipefail
MODEL=${MODEL:-.cache/models/qwen3.6-27b}; SLUG=qwen3.6-27b; GPU_MEMORY=${GPU_MEMORY:-28GiB}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-artifacts}
ACTIVE=$ARTIFACT_ROOT/$SLUG/jacobian-lens
CANDIDATE_ROOT=$ACTIVE/candidates/chinese_simplified
BENCHMARK_ROOT=$CANDIDATE_ROOT/benchmarks
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=$BENCHMARK_ROOT/runs/$RUN_ID
LOG=$RUN_ROOT/runner.log
mkdir -p "$RUN_ROOT" "$BENCHMARK_ROOT/one_prompt" "$BENCHMARK_ROOT/eight_prompts" "$CANDIDATE_ROOT"
exec > >(tee -a "$LOG") 2>&1
trap 'status=$?; printf "{\"status\":\"failed\",\"exit_code\":%s}\n" "$status" > "$RUN_ROOT/failed.json"; exit "$status"' EXIT
progress() { uv run python - "$RUN_ROOT/progress.json" "$1" <<'PY'
import json,sys,datetime,os
p,stage=sys.argv[1:]; tmp=p+'.tmp'; json.dump({'stage':stage,'run_id':os.path.basename(os.path.dirname(p)),'updated_at':datetime.datetime.now(datetime.UTC).isoformat()},open(tmp,'w'),indent=2); os.replace(tmp,p)
PY
}
[[ -d "$MODEL" ]] || { echo "checkpoint missing: $MODEL"; exit 2; }
progress probe_dim1
if [[ ! -f "$RUN_ROOT/probe-dim1.json" ]]; then uv run python scripts/probe_qwen27b.py --model "$MODEL" --dim-batch 1 --gpu-memory "$GPU_MEMORY" --output "$RUN_ROOT/probe-dim1.json"; fi
if [[ "${RUN_DIM_BATCH_2:-1}" == 1 && ! -f "$RUN_ROOT/probe-dim2.json" ]]; then
  progress probe_dim2
  if ! uv run python scripts/probe_qwen27b.py --model "$MODEL" --dim-batch 2 --gpu-memory "$GPU_MEMORY" --output "$RUN_ROOT/probe-dim2.json"; then
    printf '{"status":"failed","fallback":"dim_batch_1","reason":"dim_batch_2_probe_failed"}\n' > "$RUN_ROOT/probe-dim2-failed.json"
  fi
fi
uv run python scripts/prepare_qwen_calibration.py --output-dir "data/calibration/$SLUG" --model-slug "$SLUG"
uv run python scripts/prepare_qwen_lens_eval.py --output "data/evaluations/$SLUG/bilingual_intermediate_holdout.jsonl" --model-slug "$SLUG"
FIT_ARGS=(--model "$MODEL" --calibration-file "data/calibration/$SLUG/chinese_simplified.jsonl" --chat-template --skip-first 16 --max-seq-len 128 --dim-batch "${DIM_BATCH:-1}" --device-map qwen27b_two_gpu --max-memory-json "{\"0\":\"$GPU_MEMORY\",\"1\":\"$GPU_MEMORY\"}" --selection-basis inherited_qwen3.5_4b_operational_winner)
for N in 1 8; do
  NAME=$([[ "$N" == 1 ]] && echo one_prompt || echo eight_prompts)
  OUT="$BENCHMARK_ROOT/$NAME/jacobian_lens.pt"; TIMING="$BENCHMARK_ROOT/$NAME/timing.json"; VALIDATION="$BENCHMARK_ROOT/$NAME/resume_validation.json"; mkdir -p "$(dirname "$OUT")"; progress benchmark_$NAME
  if [[ ! -f "$OUT" || ! -f "$TIMING" ]]; then
    start=$(date +%s); uv run fit-jacobian-lens "${FIT_ARGS[@]}" --calibration-prompts "$N" --output "$OUT"; end=$(date +%s)
    uv run python - "$TIMING" "$N" "$start" "$end" <<'PY'
import json,sys,os
p,n,s,e=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]); t=p+'.tmp'; json.dump({'prompts':n,'elapsed_seconds':e-s,'seconds_per_prompt':(e-s)/n},open(t,'w'),indent=2); os.replace(t,p)
PY
  fi
  if [[ ! -f "$VALIDATION" ]]; then
    before=$(sha256sum "$OUT" | cut -d' ' -f1); before_meta=$(sha256sum "$OUT.metadata.json" | cut -d' ' -f1)
    uv run fit-jacobian-lens "${FIT_ARGS[@]}" --calibration-prompts "$N" --output "$OUT"
    after=$(sha256sum "$OUT" | cut -d' ' -f1); after_meta=$(sha256sum "$OUT.metadata.json" | cut -d' ' -f1)
    uv run python - "$VALIDATION" "$before" "$after" "$before_meta" "$after_meta" <<'PY'
import json,sys,os
p,b,a,bm,am=sys.argv[1:]; v={'resume_command_completed':True,'binary_hash_before':b,'binary_hash_after':a,'binary_hash_match':b==a,'metadata_hash_before':bm,'metadata_hash_after':am,'metadata_hash_changed':bm!=am}; t=p+'.tmp'; json.dump(v,open(t,'w'),indent=2); os.replace(t,p); raise SystemExit(0 if v['binary_hash_match'] else 2)
PY
  fi
  uv run python scripts/check_qwen_benchmark.py --lens "$OUT" --prompts "$N" --output "$BENCHMARK_ROOT/$NAME/quality.json"
done
uv run python - "$RUN_ROOT/gate.json" "$BENCHMARK_ROOT/eight_prompts/timing.json" "$BENCHMARK_ROOT/one_prompt/quality.json" "$BENCHMARK_ROOT/eight_prompts/quality.json" "$BENCHMARK_ROOT/one_prompt/resume_validation.json" "$BENCHMARK_ROOT/eight_prompts/resume_validation.json" "$RUN_ROOT/probe-dim1.json" <<'PY'
import json,sys,os
from llm_bias.lens_fitting.benchmark import estimate_from_benchmark,gate_benchmark
t=json.load(open(sys.argv[2])); q1=json.load(open(sys.argv[3])); q8=json.load(open(sys.argv[4])); r1=json.load(open(sys.argv[5])); r8=json.load(open(sys.argv[6])); probe=json.load(open(sys.argv[7])); finite=all(q1.get(k) and q8.get(k) for k in ('lens_finite','complete_63_layers_d5120','metadata_ok')); resume=all(r.get('resume_command_completed') and r.get('binary_hash_match') for r in (r1,r8)); stable=q1.get('stable_resources') and q8.get('stable_resources') and bool(probe.get('diagnostics',{}).get('parameter_bytes_by_device')); g=gate_benchmark(estimate_from_benchmark(elapsed_seconds=t['elapsed_seconds'],prompts=t['prompts'],evaluation_seconds=3600,promotion_seconds=60,pilot_seconds=3600),finite=finite,resume_ok=resume,stable=stable); json.dump(g,open(sys.argv[1],'w'),indent=2); print(json.dumps(g,indent=2)); raise SystemExit(0 if g['status']=='passed' else 3)
PY
CANDIDATE="$CANDIDATE_ROOT/jacobian_lens.pt"
uv run fit-jacobian-lens "${FIT_ARGS[@]}" --calibration-prompts 128 --checkpoint-every 4 --output "$CANDIDATE"
EVAL="$CANDIDATE_ROOT/evaluation.json"
uv run python scripts/evaluate_qwen_lens_single.py --model "$MODEL" --lens "$CANDIDATE" --holdout "data/evaluations/$SLUG/bilingual_intermediate_holdout.jsonl" --output "$EVAL"
uv run python scripts/promote_qwen_lens_candidate.py --model "$MODEL" --evaluation "$EVAL" --single-candidate "$CANDIDATE"
printf '{"status":"passed","run_root":"%s","canonical":"%s"}\n' "$RUN_ROOT" "$ACTIVE/jacobian_lens.pt" > "$RUN_ROOT/completed.json"
printf '{"canonical_ready":true,"model":"%s","lens":"%s","gate":"%s"}\n' "$SLUG" "$ACTIVE/jacobian_lens.pt" "$RUN_ROOT/gate.json" > "$ACTIVE/canonical-ready.json"
trap - EXIT
