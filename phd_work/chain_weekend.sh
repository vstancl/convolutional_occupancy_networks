#!/bin/bash
# [phd] Weekend chain: two rounds of trainings, test evals, CPU input rows, report plots.
#
# Real run (tmux):   tmux new-session -d -s phd_weekend 'bash phd_work/chain_weekend.sh'
# Dry run:           CHAIN_DRY=1 bash phd_work/chain_weekend.sh
#                    (max_it 200, validate_every 100, test_split test_smoke, out/dry_w*,
#                     logs in phd_work/logs/dry_run, results in out/dry_results, report in
#                     out/dry_report, no commits)
#
# Env: DEADLINE (default Monday 2026-11-09 06:00 local; no new training starts after it);
#      CHAIN_DRY=1 for the dry run (default DEADLINE then is 2099-12-31).
# Stops only when free disk on out/ is below 20 GB. A failed step is logged with its rc and the
# chain continues. Results are committed and pushed after each finished variant (real run only).
# All trainings use SCATTER=torch_scatter (the upstream scatter), via phd_work/run.py.

cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd)
export SCATTER=torch_scatter
export PYTHONUNBUFFERED=1

DRY=${CHAIN_DRY:-0}
if [ "$DRY" = 1 ]; then
  DEADLINE=${DEADLINE:-"2099-12-31 00:00"}
  LOGDIR=phd_work/logs/dry_run
  RESDIR=out/dry_results
  CFGDIR=phd_work/configs/weekend_dry
  OUTP=out/dry_w
  IN_PREFIX=out/dry_
  export PHD_REPORT_DIR="$REPO/out/dry_report"
else
  DEADLINE=${DEADLINE:-"2026-11-09 06:00"}
  LOGDIR=phd_work/logs
  RESDIR=phd_work/results
  CFGDIR=phd_work/configs/weekend
  OUTP=out/phd_w
  IN_PREFIX=out/phd_
fi
DL_EPOCH=$(date -d "$DEADLINE" +%s) || { echo "bad DEADLINE: $DEADLINE"; exit 1; }
mkdir -p "$LOGDIR" "$RESDIR" "$CFGDIR" out
CHAIN_LOG=$LOGDIR/chain_weekend.log

log() { echo "$(date -Iseconds) $*" | tee -a "$CHAIN_LOG"; }

free_gb() { df -BG --output=avail out | tail -1 | tr -dc '0-9'; }

check_disk() {
  local f
  f=$(free_gb)
  log "disk free on out/: ${f} GB"
  if [ "${f:-0}" -lt 20 ]; then
    log "STOP: free disk below 20 GB"
    exit 2
  fi
}

# run_step <name> <ok rc list> <logfile> <cmd...>; sets RC; never stops the chain
run_step() {
  local name=$1 ok=$2 logf=$3
  shift 3
  local t0 wall status
  t0=$(date +%s)
  log "START $name"
  "$@" > "$logf" 2>&1
  RC=$?
  wall=$(( $(date +%s) - t0 ))
  status=ok
  case " $ok " in *" $RC "*) ;; *) status=FAILED ;; esac
  log "END $name rc=$RC ($status) wall=${wall}s log=$logf"
  return 0
}

# publish <message>: commit and push the chain's results (real run only)
publish() {
  [ "$DRY" = 1 ] && return 0
  local p files=()
  shopt -s nullglob
  for p in "$CHAIN_LOG" "$LOGDIR"/w*_*.log "$LOGDIR"/w*_gpu.csv "$LOGDIR"/input_row_xz_L*.log \
           "$LOGDIR"/eval_input_row_xz_L*.log "$LOGDIR"/chain_weekend_*.log \
           "$RESDIR"/w*_best.csv "$RESDIR"/input_row_xz_L*.csv "$CFGDIR"/*.yaml phd_work/report; do
    [ -e "$p" ] && files+=("$p")
  done
  shopt -u nullglob
  git add -f -- "${files[@]}" || log "git add failed"
  if git diff --cached --quiet; then
    log "nothing new to commit"
    return 0
  fi
  git commit -q -m "$1" && git push -q 2>&1 | tail -1 >> "$CHAIN_LOG"
  log "published: $1 (head $(git rev-parse --short HEAD))"
}

do_variant() {
  local R=$1 v=$2
  local name=w${R}_${v}
  local outdir=${OUTP}${R}_${v}
  local tcfg=$CFGDIR/${name}.yaml
  local ecfg=$CFGDIR/${name}_test.yaml
  local tlog=$LOGDIR/${name}.log
  local gpulog=$LOGDIR/${name}_gpu.csv
  local trc gpid

  check_disk
  if [ "$(date +%s)" -ge "$DL_EPOCH" ]; then
    log "SKIP train $name: past DEADLINE $DEADLINE"
    return 0
  fi

  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > "$gpulog" 2>&1 &
  gpid=$!
  run_step "train $name" "0 3" "$tlog" uv run python phd_work/run.py train.py "$tcfg"
  trc=$RC
  kill $gpid 2>/dev/null
  wait $gpid 2>/dev/null

  if [ "$trc" = 0 ] || [ "$trc" = 3 ]; then
    check_disk
    run_step "generate $name" "0" "$LOGDIR/${name}_generate.log" \
      uv run python phd_work/run.py generate.py "$ecfg"
    run_step "eval $name" "0" "$LOGDIR/${name}_eval.log" \
      uv run python phd_work/run.py eval_meshes.py "$ecfg"
    if [ -f "$outdir/generation_best/eval_meshes.csv" ]; then
      cp "$outdir/generation_best/eval_meshes.csv" "$RESDIR/${name}_best.csv"
      log "copied $outdir/generation_best/eval_meshes.csv -> $RESDIR/${name}_best.csv"
    else
      log "no eval CSV for $name"
    fi
  else
    log "eval skipped for $name: training rc=$trc (no finished model)"
  fi
  log "disk used by $outdir: $(du -sh "$outdir" 2>/dev/null | cut -f1)"
  publish "[phd] weekend R${R} ${v}: train rc=${trc}, logs and results"
}

# ---- start ----------------------------------------------------------------------
log "=== chain_weekend start (dry=$DRY, deadline $DEADLINE, scatter $SCATTER)"
log "versions: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | tr '\n' ' ')"
run_step "versions torch/CUDA/cuDNN" "0" "$LOGDIR/chain_weekend_versions.log" \
  uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())"
log "versions: $(tail -1 "$LOGDIR/chain_weekend_versions.log")"
log "git: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
log "df out/: $(df -h out | tail -1)"
run_step "scatter setup check" "0" "$LOGDIR/chain_weekend_scatter_setup.log" \
  uv run python phd_work/run.py phd_work/scatter_setup_check.py
log "scatter setup: $(tail -1 "$LOGDIR/chain_weekend_scatter_setup.log")"
run_step "parity compare (cuda, SCATTER=torch_scatter; mismatch does not stop the chain)" "0" \
  "$LOGDIR/chain_weekend_parity.log" uv run python phd_work/run.py phd_work/parity.py compare --device cuda
log "parity: $(grep -E 'PARITY' "$LOGDIR/chain_weekend_parity.log" | tail -1) (rc=$RC)"

uv run python phd_work/weekend_configs.py --round 1 --cfg-dir "$CFGDIR" $([ "$DRY" = 1 ] && echo --dry) >> "$CHAIN_LOG" 2>&1
uv run python phd_work/weekend_configs.py --round 2 --cfg-dir "$CFGDIR" $([ "$DRY" = 1 ] && echo --dry) >> "$CHAIN_LOG" 2>&1

# timeline estimate: training = max_it * s/it + ~0.45 h validation; eval (generate + eval) ~0.8 h each.
# s/it (torch_scatter, from the dry-run timing runs and the earlier full runs; see port-log P3):
#   voxel3p 0.1084 (3cls run, torch_scatter), dexel3p_L32 0.1007 (P7 chain), dexel3p_L2 0.0775,
#   dexel1p_L32 0.0740, dexel1p_L2 0.0530, voxel1p 0.0440, dexel3p_L4 0.0790, grid32 0.3530, dexel3p_L1 0.0780
declare -A SPI=( [voxel3p]=0.1084 [dexel3p_L32]=0.1007 [dexel3p_L2]=0.0775 [dexel1p_L32]=0.0740
                 [dexel1p_L2]=0.0530 [voxel1p]=0.0440 [dexel3p_L4]=0.0790 [grid32]=0.3530 [dexel3p_L1]=0.0780 )
if [ "$DRY" != 1 ]; then
  total=0
  for R in 1 2; do
    if [ "$R" = 1 ]; then vs="voxel3p dexel3p_L32 dexel3p_L2 dexel1p_L32 dexel1p_L2 voxel1p dexel3p_L4 grid32 dexel3p_L1"
    else vs="voxel3p dexel3p_L32 dexel3p_L2 dexel1p_L32 dexel1p_L2 voxel1p"; fi
    rt=0
    for v in $vs; do
      h=$(awk -v s="${SPI[$v]}" 'BEGIN{printf "%.2f", 120000*s/3600 + 0.45 + 0.8}')
      log "estimate R$R $v: ~${h} h (train + eval; s/it ${SPI[$v]})"
      rt=$(awk -v a="$rt" -v b="$h" 'BEGIN{print a+b}')
    done
    log "estimate round $R total: ~$(awk -v t="$rt" 'BEGIN{printf "%.1f", t}') h"
    total=$(awk -v a="$total" -v b="$rt" 'BEGIN{print a+b}')
  done
  left=$(( (DL_EPOCH - $(date +%s)) / 3600 ))
  log "estimate both rounds: ~$(awk -v t="$total" 'BEGIN{printf "%.1f", t}') h; hours to deadline: ${left}"
else
  log "dry run: max_it 200 (no timeline estimate)"
fi

for R in 1 2; do
  if [ "$R" = 1 ]; then
    vs="voxel3p dexel3p_L32 dexel3p_L2 dexel1p_L32 dexel1p_L2 voxel1p dexel3p_L4 grid32 dexel3p_L1"
  else
    vs="voxel3p dexel3p_L32 dexel3p_L2 dexel1p_L32 dexel1p_L2 voxel1p"
  fi
  for v in $vs; do do_variant "$R" "$v"; done
done

# ---- after the last training (or the deadline): CPU steps -----------------------
for L in 2 32; do
  check_disk
  run_step "input_row xz L=$L" "0" "$LOGDIR/input_row_xz_L${L}.log" \
    uv run python phd_work/input_row.py --L "$L" --split test --planes xz --out-dir "${IN_PREFIX}input_xz_L${L}"
  run_step "eval input_row xz L=$L" "0" "$LOGDIR/eval_input_row_xz_L${L}.log" \
    uv run python eval_meshes.py "$CFGDIR/input_xz_L${L}.yaml"
  if [ -f "${IN_PREFIX}input_xz_L${L}/generation/eval_meshes.csv" ]; then
    cp "${IN_PREFIX}input_xz_L${L}/generation/eval_meshes.csv" "$RESDIR/input_row_xz_L${L}.csv"
  fi
done
publish "[phd] weekend: input rows for one plane (xz), L=2 and 32: logs and results"

run_step "report_plots" "0" "$LOGDIR/report_plots_weekend.log" uv run python phd_work/report_plots.py
publish "[phd] weekend: report plots and summary"
log "CHAIN DONE"
