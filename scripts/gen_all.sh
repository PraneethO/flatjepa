#!/usr/bin/env bash
# Batch trajectory generation across forest types and seeds.
#
# Lives in the repo (not /tmp) deliberately: an earlier version was written to a
# session scratchpad and was destroyed, along with all its logs, when the machine
# rebooted mid-run.
#
# Survives SSH disconnect when launched via scripts/gen_launch.sh, which wraps
# this in setsid + nohup.
#
# Usage:
#   scripts/gen_all.sh                  # default sweep
#   FOREST_TYPES="0 1" SEEDS="101 202" N_PER_SEED=20 scripts/gen_all.sh
#
# Environment:
#   POLYFLY_REPO   path to the polyfly_ral checkout (default ~/Desktop/polyfly_ral)
#   FOREST_TYPES   space-separated forest types      (default "0 2" -- see note)
#   SEEDS          space-separated base seeds        (default "101 202 303 404")
#   N_PER_SEED     forests per invocation            (default 40)
#   TARGET_CSVS    stop early once this many exist   (default 0 = no limit)

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLYFLY_REPO="${POLYFLY_REPO:-$HOME/Desktop/polyfly_ral}"
SHIM_DIR="$REPO_ROOT/scripts/shim"
LOG_DIR="$REPO_ROOT/logs"
# Only 0 (small obstacles) and 2 (large obstacles) exist. forest_params.py defines
# FOREST_SMALL_OBS id=0 and FOREST_LARGE_OBS id=2; generate_forest.py raises a bare
# Exception for anything else, per trajectory, and still exits 0 -- so passing type 1
# burns a whole phase reporting "0/360 succeeded" and looks like a real run.
FOREST_TYPES="${FOREST_TYPES:-0 2}"
SEEDS="${SEEDS:-101 202 303 404}"
N_PER_SEED="${N_PER_SEED:-40}"
TARGET_CSVS="${TARGET_CSVS:-0}"

mkdir -p "$LOG_DIR"

csv_count() { find "$POLYFLY_REPO/data/csvs/forests" -name '*.csv' 2>/dev/null | wc -l; }

echo "=== generation started $(date -Is) ==="
echo "polyfly repo : $POLYFLY_REPO"
echo "forest types : $FOREST_TYPES"
echo "seeds        : $SEEDS"
echo "n per seed   : $N_PER_SEED"
echo "starting csvs: $(csv_count)"

if [ ! -f "$SHIM_DIR/sitecustomize.py" ]; then
  echo "FATAL: headless shim missing at $SHIM_DIR/sitecustomize.py" >&2
  exit 1
fi
if [ ! -f "$POLYFLY_REPO/data/params/forests/base.yaml" ]; then
  echo "FATAL: $POLYFLY_REPO/data/params/forests/base.yaml missing." >&2
  echo "       generate_forest.py requires it and upstream does not ship it." >&2
  exit 1
fi

HSL_ARGS=()
if [ -n "${LIBHSL_DIR:-}" ]; then
  HSL_ARGS=(-v "$LIBHSL_DIR:$LIBHSL_DIR:ro" -e "LIBHSL_DIR=$LIBHSL_DIR")
  echo "HSL          : $LIBHSL_DIR (MA57, ~3x faster)"
else
  echo "HSL          : not set, using default IPOPT"
fi

for FT in $FOREST_TYPES; do
  if [ "$FT" != "0" ] && [ "$FT" != "2" ]; then
    echo "WARNING: forest type $FT does not exist upstream (only 0 and 2). Skipping." >&2
    continue
  fi
  for SEED in $SEEDS; do
    if [ "$TARGET_CSVS" -gt 0 ] && [ "$(csv_count)" -ge "$TARGET_CSVS" ]; then
      echo "=== target $TARGET_CSVS reached, stopping ==="
      break 2
    fi

    LOG="$LOG_DIR/gen_ft${FT}_s${SEED}.log"
    echo "=== ft=$FT seed=$SEED start $(date -Is) csvs=$(csv_count) ==="

    docker run --rm --user "$(id -u):$(id -g)" \
      -v "$POLYFLY_REPO:/workspace:rw" \
      -v "$SHIM_DIR:/shim:ro" \
      "${HSL_ARGS[@]+"${HSL_ARGS[@]}"}" \
      -e POLYFLY_DIR=/workspace \
      -e PYTHONPATH=/shim:/workspace/src \
      -e MPLBACKEND=Agg \
      -e HOME=/tmp \
      --workdir /workspace \
      poly-fly:latest \
      /opt/conda/envs/poly_fly/bin/python -m poly_fly.forest_planner.generate_forest \
        -n "$N_PER_SEED" --mp --pin --seed "$SEED" --forest-type "$FT" \
        >> "$LOG" 2>&1
    RC=$?
    echo "=== ft=$FT seed=$SEED exit=$RC done $(date -Is) csvs=$(csv_count) ==="
  done
done

echo "=== ALL DONE $(date -Is) total_csvs=$(csv_count) ==="
