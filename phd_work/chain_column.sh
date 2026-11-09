#!/bin/bash
# [phd] Control chain (P10): voxel-column encoder, 1-plane then 3-plane. Train (run.py), generate and
# eval on the test split for model_best.pt, copy the CSV, commit and push after each model.
# Start:    tmux new-session -d -s phd_column 'bash phd_work/chain_column.sh'
# Progress: tail -f phd_work/logs/chain_column.log ; tail -n 2 phd_work/logs/column_1plane.log
cd "$(dirname "$0")/.." || exit 1
export SCATTER=torch_scatter PYTHONUNBUFFERED=1
LOGDIR=phd_work/logs
CHAIN_LOG=$LOGDIR/chain_column.log

log() { echo "$(date -Iseconds) $*" | tee -a "$CHAIN_LOG"; }

check_disk() {
  local f
  f=$(df -BG --output=avail out | tail -1 | tr -dc '0-9')
  log "disk free on out/: ${f} GB"
  if [ "${f:-0}" -lt 20 ]; then log "STOP: free disk below 20 GB"; exit 2; fi
}

run_step() {  # run_step <name> <ok rc list> <logfile> <cmd...>; sets RC
  local name=$1 ok=$2 logf=$3
  shift 3
  local t0 status
  t0=$(date +%s)
  log "START $name"
  "$@" > "$logf" 2>&1
  RC=$?
  status=ok
  case " $ok " in *" $RC "*) ;; *) status=FAILED ;; esac
  log "END $name rc=$RC ($status) wall=$(( $(date +%s) - t0 ))s log=$logf"
}

publish() {  # publish <message>
  git add -f -- "$CHAIN_LOG" $LOGDIR/column_*.log $LOGDIR/column_*_gpu.csv phd_work/results/column_*_best.csv \
      phd_work/configs/voxel_column_*.yaml phd_work/column_encoder.py phd_work/dexel_register.py \
      phd_work/tests/test_column_encoder.py 2>/dev/null
  if git diff --cached --quiet; then log "nothing new to commit"; return 0; fi
  git commit -q -m "$1" && git push -q 2>&1 | tail -1 >> "$CHAIN_LOG"
  log "published: $1 (head $(git rev-parse --short HEAD))"
}

log "=== chain_column start (scatter torch_scatter)"
for V in 1plane 3plane; do
  name=column_${V}
  outdir=$(uv run python -I -c "
import sys,os; sys.path.insert(0,'.')
from src.config import load_config
print(load_config('phd_work/configs/voxel_column_${V}_3cls.yaml','configs/default.yaml')['training']['out_dir'])" 2>/dev/null | tail -1)
  check_disk
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > $LOGDIR/${name}_gpu.csv 2>&1 &
  gpid=$!
  run_step "train $name" "0 3" $LOGDIR/${name}.log uv run python phd_work/run.py train.py phd_work/configs/voxel_column_${V}_3cls.yaml
  trc=$RC
  kill $gpid 2>/dev/null; wait $gpid 2>/dev/null
  if [ "$trc" = 0 ] || [ "$trc" = 3 ]; then
    run_step "generate $name" "0" $LOGDIR/${name}_generate.log uv run python phd_work/run.py generate.py phd_work/configs/voxel_column_${V}_3cls_test.yaml
    run_step "eval $name" "0" $LOGDIR/${name}_eval.log uv run python phd_work/run.py eval_meshes.py phd_work/configs/voxel_column_${V}_3cls_test.yaml
    if [ -f "$outdir/generation_best/eval_meshes.csv" ]; then
      cp "$outdir/generation_best/eval_meshes.csv" phd_work/results/column_${V}_best.csv
      log "copied CSV -> phd_work/results/column_${V}_best.csv"
    fi
  else
    log "eval skipped for $name: training rc=$trc"
  fi
  log "disk used by $outdir: $(du -sh "$outdir" 2>/dev/null | cut -f1)"
  publish "[phd] control: voxel column encoder, ${V} (train rc=${trc}), logs and results"
done
log "CHAIN DONE"
