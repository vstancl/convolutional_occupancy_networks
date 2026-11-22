#!/usr/bin/env bash
# H5 pilot chain (H5-5b). One command to start the pilot:
#
#   cd ~/PhD/convonet && h5/scripts/pilot.sh            # full pilot, runs in ~/PhD/runs/h5_pilot
#   cd ~/PhD/convonet && h5/scripts/pilot.sh --dry-run  # the whole chain with small numbers, in a scratch directory
#
# Runs, in order: P1_hits_8, P1_occfrac_128 (train subset, eval on it); P2_hits_8, P2_occfrac_128,
# P2_points_strong (full train). Each run resumes from model_last.pt if it exists. The chain stops at the first
# failure (set -e). After P2: evaluation of the checkpoints on the validation shapes (all rays), the P2 decoding
# timing, and the report. Everything is logged to ROOT/chain.log.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
TRAIN="$REPO/h5/train.py"
SCRIPTS="$REPO/h5/scripts"

if [[ "${1:-}" == "--dry-run" ]]; then
  ROOT="${PILOT_DRY_ROOT:-${TMPDIR:-/tmp}/h5_pilot_dry_$(date +%Y%m%d_%H%M%S)}"
  P1_MAX=40; P1_VAL=20; P2_MAX=40; P2_VAL=20
  ITS="20,40"; P1_IT=40; LAST=40
  SUBSET=8; EVAL_SHAPES=4; TIME_SHAPES=8
  EXTRA=(--set val_shapes=4 --set val_rays=256 --set log_every=10)
  REPORT_OUT="$ROOT/report"
else
  ROOT="$HOME/PhD/runs/h5_pilot"
  P1_MAX=10000; P1_VAL=2500; P2_MAX=20000; P2_VAL=5000
  ITS="5000,10000,15000,20000"; P1_IT=10000; LAST=20000
  SUBSET=256; EVAL_SHAPES=200; TIME_SHAPES=100
  EXTRA=()
  REPORT_OUT="$REPO/h5/results/pilot"
fi
mkdir -p "$ROOT"
LOG="$ROOT/chain.log"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

run_train() {  # run_train NAME CONFIG MAX_IT VAL_EVERY [extra train.py args...]
  local name=$1 cfg=$2 max=$3 val=$4
  shift 4
  local out="$ROOT/$name"
  local resume=()
  if [[ -f "$out/model_last.pt" ]]; then
    resume=(--resume)
  fi
  log "START $name (max_it $max, val_every $val, resume ${#resume[@]})"
  "$PY" "$TRAIN" "$REPO/h5/configs/$cfg.yaml" --out-dir "$out" --max-it "$max" "${resume[@]}" \
      --set val_every="$val" "$@" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
  log "END $name"
}

log "PILOT CHAIN root=$ROOT dry=${1:-no}"
run_train P1_hits_8 hits_8 "$P1_MAX" "$P1_VAL" --set train_subset="$SUBSET" --set eval_on_train=true
run_train P1_occfrac_128 occfrac_128 "$P1_MAX" "$P1_VAL" --set train_subset="$SUBSET" --set eval_on_train=true
run_train P2_hits_8 hits_8 "$P2_MAX" "$P2_VAL"
run_train P2_occfrac_128 occfrac_128 "$P2_MAX" "$P2_VAL"
run_train P2_points_strong points_strong "$P2_MAX" "$P2_VAL"

log "START eval (checkpoints $ITS, $EVAL_SHAPES val shapes, all rays)"
"$PY" "$SCRIPTS/pilot_report.py" eval --root "$ROOT" --its "$ITS" --n-val-shapes "$EVAL_SHAPES" 2>&1 | tee -a "$LOG"
log "END eval"

log "START timing of the P2 checkpoints at $LAST (hits_8_plane and occfrac_128 untrained)"
"$PY" "$SCRIPTS/time_decode.py" --ckpt hits_8="$ROOT/P2_hits_8/model_it$LAST.pt" \
    --ckpt points_strong="$ROOT/P2_points_strong/model_it$LAST.pt" \
    --n-shapes "$TIME_SHAPES" --tag p2 --out-dir "$ROOT/timing" 2>&1 | tee -a "$LOG"
log "END timing"

log "START report"
"$PY" "$SCRIPTS/pilot_report.py" report --root "$ROOT" --its "$ITS" --p1-it "$P1_IT" --out "$REPORT_OUT" \
    2>&1 | tee -a "$LOG"
log "END report -> $REPORT_OUT"
log "CHAIN DONE"
