#!/usr/bin/env bash
# H5-K2-2 chain: point decoders with ray-interval query points (H5-K2-1a/b), Q0 memorisation settings. Start it with:
#
#   cd ~/PhD/convonet && h5/scripts/k2.sh              # the real chain, root ~/PhD/runs/k2
#   cd ~/PhD/convonet && h5/scripts/k2.sh --dry-run    # tiny numbers, scratch directory, CPU only
#
# 1. Train K2_points_rayint_a and K2_points_rayint_b from scratch with the Q0 settings: train_subset 32 (seed 0, the
#    same 32 shapes as Q0), eval_on_train, batch 32, max_it 60000, lr drop to 1e-5 at 50000, validation every 5000.
# 2. Evaluate model_it{5000..60000} of both runs on the same 32 shapes, all rays, with the H5-6 diagnostics and the
#    band selection over the iso grid (q0_report.py eval --grid -> ROOT/eval/{run}_it{N}_grid.json).
# 3. Re-evaluate Q0_points_strong model_it60000 of ~/PhD/runs/h5_q0 the same way (eval only). It writes
#    h5_q0/eval/Q0_points_strong_it60000_grid.json next to the old Q0 files, which are kept. In a dry run a tiny
#    Q0_points_strong is trained in the scratch root instead.
# 4. Write the report (k2_report.py) to h5/results/k2, or ROOT/report for dry runs.
#
# Every run resumes from model_last.pt if it exists; a finished run is skipped by train.py. Evaluations are written
# once and kept. The chain stops at the first failure (set -e). Everything is logged to ROOT/chain.log.
set -euo pipefail

DRY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    *) echo "unknown argument: $arg (use --dry-run)" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
TRAIN="$REPO/h5/train.py"
SCRIPTS="$REPO/h5/scripts"
LR_NEW="1.0e-5"  # YAML needs the dot: 1e-5 would be read as a string

if (( DRY )); then
  ROOT="${K2_DRY_ROOT:-${TMPDIR:-/tmp}/h5_k2_dry_$(date +%Y%m%d_%H%M%S)}"
  Q0_ROOT="$ROOT/q0"
  export CUDA_VISIBLE_DEVICES=""  # the GPU is busy with other runs: the dry run stays on the CPU
  SUBSET=2; BATCH=2; MAX=30; VAL=10; LR_STEP=20
  ITS="10,20,30"; Q0_IT=30
  EXTRA=(--set log_every=10 --set n_rays=32 --set n_points=1000 --set num_workers=0)
  REPORT_OUT="$ROOT/report"
else
  ROOT="$HOME/PhD/runs/k2"
  Q0_ROOT="$HOME/PhD/runs/h5_q0"
  SUBSET=32; BATCH=32; MAX=60000; VAL=5000; LR_STEP=50000
  ITS="5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000"; Q0_IT=60000
  EXTRA=()
  REPORT_OUT="$REPO/h5/results/k2"
fi
mkdir -p "$ROOT"
LOG="$ROOT/chain.log"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

run_train() {  # run_train ROOT NAME CONFIG
  local root=$1 name=$2 cfg=$3
  local out="$root/$name"
  local resume=()
  if [[ -f "$out/model_last.pt" ]]; then
    resume=(--resume)
  elif [[ -f "$out/log.jsonl" ]]; then
    # a crash before the first validation: train.py refuses to start in a directory with a log and no checkpoint
    log "ERROR $name: $out has log.jsonl but no model_last.pt; move or delete $out and start again"
    exit 1
  fi
  log "START $name (max_it $MAX, val_every $VAL, lr drop $LR_STEP -> $LR_NEW, resume ${#resume[@]})"
  "$PY" "$TRAIN" "$REPO/h5/configs/$cfg.yaml" --out-dir "$out" --max-it "$MAX" "${resume[@]}" \
      --set val_every="$VAL" --set train_subset="$SUBSET" --set eval_on_train=true --set batch_size="$BATCH" \
      --set "lr_steps=[[${LR_STEP}, ${LR_NEW}]]" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
  log "END $name"
}

log "K2 CHAIN root=$ROOT q0_root=$Q0_ROOT dry=$DRY"
run_train "$ROOT" K2_points_rayint_a points_rayint_a
run_train "$ROOT" K2_points_rayint_b points_rayint_b
if (( DRY )); then
  mkdir -p "$Q0_ROOT"
  run_train "$Q0_ROOT" Q0_points_strong points_strong  # stand-in for the real Q0 run
fi

log "START eval K2 (checkpoints $ITS, $SUBSET training shapes, all rays, grid and band selection)"
"$PY" "$SCRIPTS/q0_report.py" eval --root "$ROOT" --runs K2_points_rayint_a,K2_points_rayint_b --its "$ITS" --grid \
    2>&1 | tee -a "$LOG"
log "END eval K2"

log "START eval Q0_points_strong it $Q0_IT ($Q0_ROOT, grid and band selection; old eval files kept)"
"$PY" "$SCRIPTS/q0_report.py" eval --root "$Q0_ROOT" --runs Q0_points_strong --its "$Q0_IT" --grid 2>&1 | tee -a "$LOG"
log "END eval Q0_points_strong"

log "START report"
"$PY" "$SCRIPTS/k2_report.py" --root "$ROOT" --q0-root "$Q0_ROOT" --final-it "$Q0_IT" --out "$REPORT_OUT" \
    2>&1 | tee -a "$LOG"
log "END report -> $REPORT_OUT"
log "K2 CHAIN DONE"
