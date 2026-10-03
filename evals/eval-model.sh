#!/bin/bash
# eval-model.sh — the standard, reusable model scorecard.
#
# Runs the discriminating frontier battery against ANY endpoint and writes one
# scorecard.md + scorecard.json. Judge-free where it can be (reasoning_hard solver-
# verified, LiveCodeBench exec-graded, function_call schema-checked); claw uses an
# INDEPENDENT cloud judge (protolabs/cloud) so a local model never grades itself.
#
# Usage:
#   ./eval-model.sh <label> <target-url> <model-name> [--quick|--full]
#
# Examples:
#   ./eval-model.sh ThinkingCap http://localhost:8041/v1 reasoning            # quick (default)
#   ./eval-model.sh Ornith-35B  http://localhost:8040/v1 fast --full
#   ./eval-model.sh CoderNext   http://localhost:8032/v1 coder --quick
#
# Tiers:
#   quick (default): claw(10) + reasoning_hard + livecodebench + function_call   (~30-45 min)
#   full:            quick + instruction_following + structured_hard + safety + creative_writing
#
# Judge (override via env): JUDGE_GATEWAY_URL (default http://ava:4000/v1), JUDGE_MODEL
#   (default protolabs/cloud). Needs GATEWAY_API_KEY (sourced from evals/.env or the env).
# Knobs: EVAL_THINKING=0 disables thinking on reasoning_hard. LCB_LIMIT=N (default 30).

set -uo pipefail
SD="$(cd "$(dirname "$0")" && pwd)"; cd "$SD"

[ $# -lt 3 ] && { echo "usage: ./eval-model.sh <label> <target-url> <model-name> [--quick|--full]"; exit 1; }
LABEL="$1"; URL="$2"; MODEL="$3"; TIER="quick"
[ "${4:-}" = "--full" ] && TIER="full"

# ── venv ──
[ -d venv ] && source venv/bin/activate || { echo "no venv (run ./scripts/setup.sh)"; exit 1; }

# ── secrets: source .env for GATEWAY_API_KEY, then OVERRIDE the (stale :8000) judge vars ──
if [ -f .env ]; then set -a; source .env; set +a; fi
export JUDGE_GATEWAY_URL="${JUDGE_GATEWAY_URL_OVERRIDE:-${JUDGE_GATEWAY_URL_CLOUD:-http://ava:4000/v1}}"
export JUDGE_MODEL="${JUDGE_MODEL_CLOUD:-protolabs/cloud}"
# .env may have repinned these to a dead port — force the intended independent cloud judge:
[ -n "${JUDGE_URL:-}" ] && export JUDGE_GATEWAY_URL="$JUDGE_URL"
[ -n "${JUDGE_MDL:-}" ] && export JUDGE_MODEL="$JUDGE_MDL"
if [ -z "${GATEWAY_API_KEY:-}" ]; then
  echo "⚠️  GATEWAY_API_KEY unset — the cloud judge ($JUDGE_MODEL) will fail; claw/breadth will 0.5-fallback."
fi

TS=$(date -u +%Y%m%d_%H%M%S)
OUT="results/scorecard-${LABEL}-${TS}"; mkdir -p "$OUT"
LOG="$OUT/run.log"; : > "$LOG"
echo "=== eval-model: $LABEL  tier=$TIER  target=$MODEL@$URL  judge=$JUDGE_MODEL@$JUDGE_GATEWAY_URL ==="
echo "    out=$OUT" | tee -a "$LOG"

run() { echo -e "\n\$ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

# ── claw (agentic, independent cloud judge) ──
if [ "$TIER" = full ]; then
  CLAW_TASKS="T02,T04,T06,T08,T10_contact_lookup,T12,T14,T16,T18,T20,T22,T24,T26,T28,T30,T32,T34,T36,T38,T40,T42,T44,T46,T48,T50,T86,T87,T88,T93,T99"
else
  CLAW_TASKS="T02,T04,T06,T08,T10_contact_lookup,T14,T18,T24,T28,T38"
fi
before=$(ls -d results/*/ 2>/dev/null | sort)
run python -m runners.run_claw --model "$MODEL" --gateway-url "$URL" --tasks "$CLAW_TASKS" --port-offset 200 --workers 4
after=$(ls -d results/*/ 2>/dev/null | sort)
CLAW_DIR=$(comm -13 <(echo "$before") <(echo "$after") | grep -v "scorecard-" | head -1)
echo "    claw dir: ${CLAW_DIR:-<none>}" | tee -a "$LOG"

# ── reasoning_hard (solver-verified) ──
# ── preflight: does the served context fit the budgets we are about to request? ──
# Learned 2026-08-22 the expensive way: a lane served at --max-model-len 32768 while the
# LCB runner asks for max_tokens=32768 makes EVERY request 400 (prompt + output > window).
# The suite does not crash -- it records tokens=0 for all 30 problems and reports
# livecodebench 0.0, which is indistinguishable from "the model cannot code". reasoning_hard
# died with the same 400. Fail loudly here instead of publishing a zero.
NEED=$(( 32768 + 8192 ))
CTX=$(curl -s -m 10 "$URL/models" 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)['data'][0]
    print(d.get('max_model_len') or 0)
except Exception:
    print(-1)
" 2>/dev/null)
if [ "${CTX:-0}" -gt 0 ] && [ "$CTX" -lt "$NEED" ]; then
  echo "PREFLIGHT FAIL: served max_model_len=$CTX but this battery requests up to 32768 output"
  echo "  tokens plus prompt (need >= $NEED). Every request would 400 and the suite would"
  echo "  silently score 0.0. Re-serve with a larger --max-model-len, or lower the budgets."
  exit 1
fi
[ "${CTX:-0}" -le 0 ] && echo "note: could not read max_model_len from $URL/models — skipping preflight"

THINK=""; [ "${EVAL_THINKING:-1}" = 1 ] && THINK="--thinking"
run python -m runners.run_custom --suite reasoning_hard --model "$MODEL" --gateway-url "$URL" \
    --output-dir "$OUT/custom-reasoning_hard" $THINK

# ── livecodebench (exec-graded, contamination-resistant) ──
# Coding CAPABILITY = thinking OFF at prod budget (32k): it's the config where a model actually
# emits code. Thinking-ON is a runaway trap on our non-coder lanes (Ornith aces a problem thinking-
# off in 8s, but burns the full 32k emitting no code thinking-on) — that pathology is characterized
# separately (EVAL_LCB_THINK_PROBE), never averaged into the headline. Coder lanes are non-thinking
# natively, so --no-thinking is a no-op for them = fair, same config across lanes.
run python -m runners.run_livecodebench --model "$MODEL" --gateway-url "$URL" \
    --limit "${LCB_LIMIT:-30}" --min-date 2025-01-01 --no-thinking --max-tokens 32768 --output-dir "$OUT"

# ── function_call (schema-checked) ──
run python -m runners.run_function_call --model "$MODEL" --gateway-url "$URL" --all-suites --output-dir "$OUT"

# ── breadth (full tier only) ──
BREADTH=""
if [ "$TIER" = full ]; then
  for S in instruction_following structured_hard safety creative_writing; do
    run python -m runners.run_custom --suite "$S" --model "$MODEL" --gateway-url "$URL" \
        --output-dir "$OUT/custom-$S"
    BREADTH="${BREADTH:+$BREADTH,}$S"
  done
fi

# ── aggregate ──
echo -e "\n=== scorecard ===" | tee -a "$LOG"
python -m runners.scorecard --out-dir "$OUT" --label "$LABEL" --model "$MODEL" --url "$URL" \
    --tier "$TIER" --claw-dir "${CLAW_DIR:-}" --custom-suites "reasoning_hard${BREADTH:+,$BREADTH}" \
    --judge-log "$LOG" --judge-model "$JUDGE_MODEL" | tee -a "$LOG"
echo -e "\n→ $OUT/scorecard.md"
