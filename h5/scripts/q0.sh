#!/usr/bin/env bash
# H5-6 memorisation probe Q0. Start it with:
#
#   cd ~/PhD/convonet && h5/scripts/q0.sh              # full probe, runs in ~/PhD/runs/h5_q0 (about 3 h)
#   cd ~/PhD/convonet && h5/scripts/q0.sh --dry-run    # the whole chain with tiny numbers, in a scratch directory
#
# Runs, in order: Q0_points_strong, Q0_hits_8 (32 training shapes, all rays at evaluation, eval_on_train, batch 32,
# 30000 iterations, validation every 5000). Each run resumes from model_last.pt if it exists; a finished run is
# skipped. Then every checkpoint model_it{5000..30000}.pt is evaluated on the 32 shapes (all rays) and the report
# is written to h5/results/q0 (report.md, report.json). The chain stops at the first failure (set -e).
# Everything is logged to ROOT/chain.log.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
TRAIN="$REPO/h5/train.py"
SCRIPTS="$REPO/h5/scripts"

if [[ "${1:-}" == "--dry-run" ]]; then
  ROOT="${Q0_DRY_ROOT:-${TMPDIR:-/tmp}/h5_q0_dry_$(date +%Y%m%d_%H%M%S)}"
  MAX=40; VAL=20; ITS="20,40"; FINAL=40; PREV=20
  SUBSET=4; BATCH=2
  EXTRA=(--set log_every=10 --set n_rays=64 --set n_points=2000)
  REPORT_OUT="$ROOT/report"
else
  ROOT="$HOME/PhD/runs/h5_q0"
  MAX=30000; VAL=5000; ITS="5000,10000,15000,20000,25000,30000"; FINAL=30000; PREV=20000
  SUBSET=32; BATCH=32
  EXTRA=()
  REPORT_OUT="$REPO/h5/results/q0"
fi
mkdir -p "$ROOT"
LOG="$ROOT/chain.log"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

run_train() {  # run_train NAME CONFIG
  local name=$1 cfg=$2
  local out="$ROOT/$name"
  local resume=()
  if [[ -f "$out/model_last.pt" ]]; then
    resume=(--resume)
  elif [[ -f "$out/log.jsonl" ]]; then
    # a crash before the first validation: train.py refuses to start in a directory with a log and no checkpoint
    log "ERROR $name: $out has log.jsonl but no model_last.pt; move or delete $out and start again"
    exit 1
  fi
  log "START $name (max_it $MAX, val_every $VAL, resume ${#resume[@]})"
  "$PY" "$TRAIN" "$REPO/h5/configs/$cfg.yaml" --out-dir "$out" --max-it "$MAX" "${resume[@]}" \
      --set val_every="$VAL" --set train_subset="$SUBSET" --set eval_on_train=true --set batch_size="$BATCH" \
      "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
  log "END $name"
}

log "Q0 CHAIN root=$ROOT dry=${1:-no}"
run_train Q0_points_strong points_strong
run_train Q0_hits_8 hits_8

log "START eval (checkpoints $ITS, 32 training shapes, all rays)"
"$PY" "$SCRIPTS/q0_report.py" eval --root "$ROOT" --its "$ITS" 2>&1 | tee -a "$LOG"
log "END eval"

log "START report"
"$PY" "$SCRIPTS/q0_report.py" report --root "$ROOT" --its "$ITS" --final-it "$FINAL" --prev-it "$PREV" \
    --out "$REPORT_OUT" 2>&1 | tee -a "$LOG"
log "END report -> $REPORT_OUT"
log "Q0 CHAIN DONE"
