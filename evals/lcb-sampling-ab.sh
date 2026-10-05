#!/usr/bin/env bash
# Graded LiveCodeBench A/B: does our harness sampling (temp 0.2) depress the
# Ornith-1.5 score vs the model's documented coding setting (temp 0.6)?
#
# 3 trials per arm — LCB is single-trial-noisy (8-14/30 on IDENTICAL weights),
# so a delta under ~0.10 means nothing without repetition.
set -u
LAB=/home/ava/dev/lab
PY=$LAB/.venv/bin/python
OUTROOT=$LAB/evals/results/lcb-sampling-ab-$(date +%Y%m%d_%H%M%S)
PORT=8062
MODEL=ornith-1.5-9b-nvfp4
mkdir -p "$OUTROOT"

echo "serving Ornith-1.5-9B-NVFP4 on GPU0 :$PORT (util 0.16, sized against ~20GiB free)"
UTIL=0.17 MAXLEN=40960 PORT=$PORT GPU=0 bash "$LAB/models/serve-ornith15-9b-nvfp4.sh" \
  > "$OUTROOT/serve.log" 2>&1 &
SRV=$!
for i in $(seq 1 240); do
  sleep 5
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
done
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "SERVE FAILED"; tail -30 "$OUTROOT/serve.log"; kill $SRV 2>/dev/null; exit 1
fi
# gate on a real completion, not /v1/models — /health lies when the engine wedges
# Gate at the ACTUAL budget the eval uses. A 'say ok' probe at max_tokens=2048
# passes even when every real request 400s because max-model-len cannot hold
# prompt + 32768 output — which is exactly how the first attempt scored 0.000
# across 6 runs in 10s each.
PROBE=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"say ok\"}],\"max_tokens\":32768}")
if echo "$PROBE" | grep -q '"error"'; then
  echo "PREFLIGHT FAILED at the eval's real budget:"; echo "$PROBE" | head -c 500
  kill $SRV 2>/dev/null; exit 1
fi
echo "preflight OK at max_tokens=32768"

for TEMP in 0.2 0.6; do
  for TRIAL in 1 2 3; do
    D="$OUTROOT/temp${TEMP}-trial${TRIAL}"
    mkdir -p "$D"
    echo "=== temp=$TEMP trial=$TRIAL  $(date +%H:%M:%S) ==="
    LCB_TEMPERATURE=$TEMP PYTHONPATH=$LAB/evals \
      "$PY" "$LAB/evals/runners/run_livecodebench.py" \
        --model "$MODEL" --gateway-url "http://127.0.0.1:$PORT/v1" --api-key dummy \
        --limit 30 --difficulty hard --max-tokens 32768 --no-thinking \
        --output-dir "$D" > "$D/run.log" 2>&1
    "$PY" - "$D" <<'PYEOF'
import json,sys,glob
f=glob.glob(sys.argv[1]+"/*.json")
if not f: print("   NO RESULT"); raise SystemExit
j=json.load(open(f[0])); s=j["summary"]
caps=sum(1 for p in j["problems"] if (p["tokens"] or 0)>=32000)
print(f"   mean={s['mean_score']:.3f} solved={s['fully_solved']} "
      f"avg_tok={s['avg_tokens']} capped={caps}/30 no_code={s['no_code_in_budget']}")
PYEOF
  done
done
kill $SRV 2>/dev/null; sleep 10
echo "AB DONE -> $OUTROOT"
