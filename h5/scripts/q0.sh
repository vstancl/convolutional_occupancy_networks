#!/usr/bin/env bash
# H5-6 memorisation probe Q0. Start it with:
#
#   cd ~/PhD/convonet && h5/scripts/q0.sh                 # v1: Q0_points_strong, Q0_hits_8 to 30000 (about 3 h)
#   cd ~/PhD/convonet && h5/scripts/q0.sh --extend        # H5-6c: resume both to 60000 with a learning-rate drop
#                                                         # at 50000 (1e-5), train Q0_occfrac_128 from scratch on the
#                                                         # same schedule; needs the v1 runs (or runs them first)
#   cd ~/PhD/convonet && h5/scripts/q0.sh --dry-run       # v1 with tiny numbers, scratch directory
#   cd ~/PhD/convonet && h5/scripts/q0.sh --extend --dry-run   # the whole extension with tiny numbers, scratch directory
#
# Every run resumes from model_last.pt if it exists; a finished run is skipped. The chain stops at the first failure
# (set -e). Everything is logged to ROOT/chain.log. The evaluation of each checkpoint is written once and kept
# (q0_report.py eval). The report is written to h5/results/q0 (report.md, report.json), or ROOT/report for dry runs.
set -euo pipefail

EXTEND=0
DRY=0
for arg in "$@"; do
  case "$arg" in
    --extend) EXTEND=1 ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown argument: $arg (use --dry-run and/or --extend)" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
TRAIN="$REPO/h5/train.py"
SCRIPTS="$REPO/h5/scripts"
LR_NEW="1.0e-5"  # YAML needs the dot: 1e-5 would be read as a string

if (( DRY )); then
  ROOT="${Q0_DRY_ROOT:-${TMPDIR:-/tmp}/h5_q0_dry_$(date +%Y%m%d_%H%M%S)}"
  SUBSET=4; BATCH=2
  EXTRA=(--set log_every=10 --set n_rays=64 --set n_points=2000)
  REPORT_OUT="$ROOT/report"
  if (( EXTEND )); then
    V1_MAX=40; EXT_MAX=80; VAL=10; LR_STEP=60
    ITS_V1="10,20,30,40"; ITS_EXT="10,20,30,40,50,60,70,80"
    FINAL_V1=40; PREV_V1=20; V2_FINAL=80
  else
    V1_MAX=40; EXT_MAX=80; VAL=20; LR_STEP=60
    ITS_V1="20,40"; ITS_EXT="20,40"
    FINAL_V1=40; PREV_V1=20; V2_FINAL=80
  fi
else
  ROOT="$HOME/PhD/runs/h5_q0"
  SUBSET=32; BATCH=32; VAL=5000
  EXTRA=()
  REPORT_OUT="$REPO/h5/results/q0"
  V1_MAX=30000; EXT_MAX=60000; LR_STEP=50000
  ITS_V1="5000,10000,15000,20000,25000,30000"
  ITS_EXT="5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000"
  FINAL_V1=30000; PREV_V1=20000; V2_FINAL=60000
fi
if (( EXTEND )); then ITS="$ITS_EXT"; else ITS="$ITS_V1"; fi
mkdir -p "$ROOT"
LOG="$ROOT/chain.log"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

run_train() {  # run_train NAME CONFIG MAX [lr]: "lr" sets the learning-rate drop (LR_STEP -> LR_NEW)
  local name=$1 cfg=$2 max=$3 lr=${4:-}
  local out="$ROOT/$name"
  local resume=() extra=()
  if [[ -f "$out/model_last.pt" ]]; then
    resume=(--resume)
  elif [[ -f "$out/log.jsonl" ]]; then
    # a crash before the first validation: train.py refuses to start in a directory with a log and no checkpoint
    log "ERROR $name: $out has log.jsonl but no model_last.pt; move or delete $out and start again"
    exit 1
  fi
  if [[ "$lr" == "lr" ]]; then
    extra=(--set "lr_steps=[[${LR_STEP}, ${LR_NEW}]]")
  fi
  log "START $name (max_it $max, val_every $VAL, resume ${#resume[@]}, lr drop ${lr:+$LR_STEP -> $LR_NEW})"
  "$PY" "$TRAIN" "$REPO/h5/configs/$cfg.yaml" --out-dir "$out" --max-it "$max" "${resume[@]}" \
      --set val_every="$VAL" --set train_subset="$SUBSET" --set eval_on_train=true --set batch_size="$BATCH" \
      "${extra[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
  log "END $name"
}

log "Q0 CHAIN root=$ROOT dry=$DRY extend=$EXTEND"
run_train Q0_points_strong points_strong "$V1_MAX"
run_train Q0_hits_8 hits_8 "$V1_MAX"
if (( EXTEND )); then
  run_train Q0_points_strong points_strong "$EXT_MAX" lr
  run_train Q0_hits_8 hits_8 "$EXT_MAX" lr
  run_train Q0_occfrac_128 occfrac_128 "$EXT_MAX" lr
fi

log "START eval (checkpoints $ITS, 32 training shapes, all rays)"
"$PY" "$SCRIPTS/q0_report.py" eval --root "$ROOT" --its "$ITS" 2>&1 | tee -a "$LOG"
log "END eval"

log "START report"
"$PY" "$SCRIPTS/q0_report.py" report --root "$ROOT" --its "$ITS" --final-it "$FINAL_V1" --prev-it "$PREV_V1" \
    --v2-final "$V2_FINAL" --lr-step "$LR_STEP" --out "$REPORT_OUT" 2>&1 | tee -a "$LOG"
log "END report -> $REPORT_OUT"
log "Q0 CHAIN DONE"
