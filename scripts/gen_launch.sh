#!/usr/bin/env bash
# Launch gen_all.sh detached so it survives SSH disconnect, terminal close, and
# the parent shell exiting. It does NOT survive a reboot -- see PROGRESS.md, an
# earlier run was lost to exactly that.
#
#   scripts/gen_launch.sh          # start
#   scripts/gen_launch.sh status   # is it running, how many trajectories
#   scripts/gen_launch.sh stop     # stop it and any planner containers
#
# All arguments/env vars accepted by gen_all.sh are passed through.

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLYFLY_REPO="${POLYFLY_REPO:-$HOME/Desktop/polyfly_ral}"
LOG_DIR="$REPO_ROOT/logs"
PIDFILE="$LOG_DIR/gen_all.pid"
DRIVER_LOG="$LOG_DIR/gen_all.log"

mkdir -p "$LOG_DIR"
csv_count() { find "$POLYFLY_REPO/data/csvs/forests" -name '*.csv' 2>/dev/null | wc -l; }

running() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

case "${1:-start}" in
  status)
    if running; then echo "RUNNING (pid $(cat "$PIDFILE"))"; else echo "NOT RUNNING"; fi
    echo "forest trajectories: $(csv_count)"
    echo "containers: $(docker ps -q | wc -l)"
    echo "--- last driver log lines ---"
    tail -5 "$DRIVER_LOG" 2>/dev/null || echo "(no driver log)"
    ;;
  stop)
    if running; then
      kill "$(cat "$PIDFILE")" 2>/dev/null && echo "stopped driver"
    else
      echo "driver not running"
    fi
    # The driver blocks on docker run; kill any surviving planner containers too.
    for c in $(docker ps -q --filter ancestor=poly-fly:latest); do
      docker kill "$c" >/dev/null 2>&1 && echo "killed container $c"
    done
    rm -f "$PIDFILE"
    ;;
  start)
    if running; then
      echo "already running (pid $(cat "$PIDFILE")); use 'stop' first"
      exit 1
    fi
    shift 2>/dev/null || true
    setsid nohup "$REPO_ROOT/scripts/gen_all.sh" "$@" >> "$DRIVER_LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "launched detached, pid $(cat "$PIDFILE")"
    echo "driver log: $DRIVER_LOG"
    echo "check with: scripts/gen_launch.sh status"
    ;;
  *)
    echo "usage: $0 [start|status|stop]" >&2; exit 2 ;;
esac
