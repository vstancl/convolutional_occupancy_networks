#!/usr/bin/env bash
# H5-K2-3 chain: points_mix_a (half near-surface as points_strong, half ray-interval variant a; K2') on the full train
# split, with the settings of the K1-1 points_strong run (only the target differs): batch 32, 120000 iterations,
# learning rate 10x lower (1e-5) from 100000, validation every 10000 (first 200 val shapes, 4096 fixed rays, band
# selection, as default.yaml). Then checkpoints 40000, 80000 and 120000 are evaluated as points_strong was in K1-1: the
# first 200 val shapes, all rays, the setting chosen by the band selection of training (q0_report.py eval,
# ROOT/eval/{run}_it{N}.json), and the K2-3 report is written against the K1 references (k1_1_report.py --k2-3; the
# K1-1 and K1-4 roots are only read, nothing there is retrained or re-evaluated). Start it with:
#
#   cd ~/PhD/convonet && h5/scripts/k2_3.sh               # the real chain, root ~/PhD/runs/k2_3
#   cd ~/PhD/convonet && h5/scripts/k2_3.sh --dry-run     # tiny numbers, scratch directory, CPU only
#
# The run resumes from model_last.pt if it exists; a finished run is skipped. The chain stops at the first failure
# (set -e). Everything is logged to ROOT/chain.log. The report goes to h5/results/k2_3 (dry runs: ROOT/report; the
# reference rows are then still the K1 rows at 40000/80000/120000).
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
K1_1_ROOT="$HOME/PhD/runs/k1_1"  # read only
K1_4_ROOT="$HOME/PhD/runs/k1_4"  # read only
REF_ITS="40000,80000,120000"

if (( DRY )); then
  ROOT="${K2_3_DRY_ROOT:-${TMPDIR:-/tmp}/h5_k2_3_dry_$(date +%Y%m%d_%H%M%S)}"
  export CUDA_VISIBLE_DEVICES=""  # the dry run stays on the CPU
  MAX=30; VAL=10; LR_STEP=20; ITS="10,20,30"; N_SHAPES=2
  EXTRA=(--set log_every=10 --set n_rays=64 --set n_points=2000 --set batch_size=2 --set train_subset=4
         --set val_shapes=2 --set val_rays=64 --set num_workers=0)
  REPORT_OUT="$ROOT/report"
else
  ROOT="$HOME/PhD/runs/k2_3"
  MAX=120000; VAL=10000; LR_STEP=100000; ITS="40000,80000,120000"; N_SHAPES=200
  EXTRA=()
  REPORT_OUT="$REPO/h5/results/k2_3"
fi
RUN="K2_3_points_mix_a"
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

log "K2-3 CHAIN root=$ROOT dry=$DRY"
run_train "$RUN" points_mix_a

log "START eval (checkpoints $ITS, val split first $N_SHAPES shapes, all rays, setting of training)"
"$PY" "$SCRIPTS/q0_report.py" eval --root "$ROOT" --runs "$RUN" --its "$ITS" --split val --n-shapes "$N_SHAPES" \
    2>&1 | tee -a "$LOG"
log "END eval"

log "START report (references of $K1_1_ROOT and $K1_4_ROOT at $REF_ITS, read only)"
"$PY" "$SCRIPTS/k1_1_report.py" --k2-3 "$ROOT" --root "$K1_1_ROOT" --root "$K1_4_ROOT" --its "$REF_ITS" \
    --new-its "$ITS" --out "$REPORT_OUT" 2>&1 | tee -a "$LOG"
log "END report -> $REPORT_OUT"
log "K2-3 CHAIN DONE"
