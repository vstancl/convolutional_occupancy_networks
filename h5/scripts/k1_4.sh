#!/usr/bin/env bash
# H5-K1-4 chain: the remaining four decoders of the first round (tsdf_128, binary_128, hits_8_plane, points_std), with
# the K1-1 settings: full train split, batch 32, 120000 iterations, learning-rate drop to 1e-5 at 100000, validation
# every 10000 (first 200 val shapes, 4096 fixed rays, band selection, as default.yaml). Then every run's checkpoints
# 40000, 80000 and 120000 are evaluated on the first 200 val shapes with all rays, over the whole decoding grid of its
# target with the band selection and the H5-6 diagnostics (q0_report.py eval --grid, as K1-1b; written once and kept),
# and the combined K1 report of the K1-1 and K1-4 roots is written. Start it with:
#
#   cd ~/PhD/convonet && h5/scripts/k1_4.sh               # the real chain, root ~/PhD/runs/k1_4
#   cd ~/PhD/convonet && h5/scripts/k1_4.sh --dry-run     # the whole chain with tiny numbers, scratch directory
#
# Every run resumes from model_last.pt if it exists; a finished run is skipped. The chain stops at the first failure
# (set -e). Everything is logged to ROOT/chain.log. The report (k1_1_report.py with --root K1_1_ROOT --root ROOT) is
# written to h5/results/k1 (report.md, report.json), or ROOT/report for dry runs. The K1-1 runs are only read.
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
K1_1_ROOT="$HOME/PhD/runs/k1_1"  # read only: its evaluations enter the combined report

if (( DRY )); then
  ROOT="${K1_DRY_ROOT:-${TMPDIR:-/tmp}/h5_k1_4_dry_$(date +%Y%m%d_%H%M%S)}"
  MAX=30; VAL=10; LR_STEP=20; ITS="10,20,30"; N_SHAPES=2
  EXTRA=(--set log_every=10 --set n_rays=64 --set n_points=2000 --set batch_size=2 --set train_subset=4
         --set val_shapes=2 --set val_rays=64)
  REPORT_OUT="$ROOT/report"
else
  ROOT="$HOME/PhD/runs/k1_4"
  MAX=120000; VAL=10000; LR_STEP=100000; ITS="40000,80000,120000"; N_SHAPES=200
  EXTRA=()
  REPORT_OUT="$REPO/h5/results/k1"
fi
RUNS="tsdf_128,binary_128,hits_8_plane,points_std"
mkdir -p "$ROOT"
LOG="$ROOT/chain.log"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

run_train() {  # run_train NAME CONFIG: one run with the K1-1 schedule
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
  log "START $name (max_it $MAX, val_every $VAL, lr drop $LR_STEP -> $LR_NEW, resume ${#resume[@]})"
  "$PY" "$TRAIN" "$REPO/h5/configs/$cfg.yaml" --out-dir "$out" --max-it "$MAX" "${resume[@]}" \
      --set val_every="$VAL" --set lr_steps="[[${LR_STEP}, ${LR_NEW}]]" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
  log "END $name"
}

log "K1-4 CHAIN root=$ROOT dry=$DRY"
run_train tsdf_128 tsdf_128
run_train binary_128 binary_128
run_train hits_8_plane hits_8_plane
run_train points_std points_std

log "START eval (checkpoints $ITS, val split first $N_SHAPES shapes, all rays, decoding grid and band selection)"
"$PY" "$SCRIPTS/q0_report.py" eval --root "$ROOT" --runs "$RUNS" --its "$ITS" --split val --n-shapes "$N_SHAPES" \
    --grid 2>&1 | tee -a "$LOG"
log "END eval"

log "START report (roots $K1_1_ROOT and $ROOT)"
"$PY" "$SCRIPTS/k1_1_report.py" --root "$K1_1_ROOT" --root "$ROOT" --its "$ITS" --out "$REPORT_OUT" 2>&1 | tee -a "$LOG"
log "END report -> $REPORT_OUT"
log "K1-4 CHAIN DONE"
