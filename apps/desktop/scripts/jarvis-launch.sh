#!/usr/bin/env bash
# Idempotent launcher for the JARVIS (Hermes) desktop dev session on WSL.
#
#   jarvis-launch.sh            start the dev session, or bring the running
#                               window to the front if one is already up
#   jarvis-launch.sh --restart  stop the running dev session cleanly, then start
#   jarvis-launch.sh --stop     stop the running dev session cleanly
#
# Why this exists: a second `npm run dev` always fails on port 5174 (vite runs
# with strictPort) and `concurrently -k` then kills that run's Electron before
# the app's single-instance lock can hand off to the existing window. Here a
# running app is focused by launching Electron once more (second-instance →
# restore + show + focus, then the extra process exits).
#
# The session runs as the systemd user unit `jarvis-desktop`: processes started
# from a `wsl.exe`/`wslg.exe` shortcut are killed when that invocation exits,
# even with setsid/nohup, but a user unit survives. The unit name also guards
# against starting the session twice. Output goes to ~/.hermes/logs.
set -u

DESKTOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV_URL="http://127.0.0.1:5174"
UNIT="jarvis-desktop"
LOG_DIR="${HERMES_HOME:-$HOME/.hermes}/logs"
LOG="$LOG_DIR/jarvis-launch.log"
ELECTRON_BIN="$DESKTOP_DIR/node_modules/electron/dist/electron"

mkdir -p "$LOG_DIR"

# Shortcut launches get a bare environment; make node/npm and WSLg resolvable.
export PATH="$HOME/.hermes/node/bin:$HOME/.local/bin:$PATH"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export PULSE_SERVER="${PULSE_SERVER:-unix:/mnt/wslg/PulseServer}"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >>"$LOG"; }

# Main Electron process of this checkout's desktop app (not helpers/zygotes).
app_pid() { pgrep -f -x "$ELECTRON_BIN \\." | head -n1; }

unit_active() { systemctl --user -q is-active "$UNIT" 2>/dev/null; }

dev_server_up() { curl -fsS -o /dev/null --max-time 2 "$DEV_URL/@vite/client" 2>/dev/null; }

port_busy() { ss -ltn 2>/dev/null | grep -q '127.0.0.1:5174 '; }

electron_env() {
  env XCURSOR_SIZE=24 HERMES_DESKTOP_DEV_SERVER="$DEV_URL" "$@"
}

# Run a command as the detached user unit, carrying the GUI/audio/WSL
# environment the user manager does not have. The session-specific
# WSL_INTEROP socket dies with the launching shell, so use init's.
start_unit() {
  systemctl --user reset-failed "$UNIT" >/dev/null 2>&1
  systemd-run --user --collect --quiet --unit="$UNIT" \
    --working-directory="$DESKTOP_DIR" \
    -p StandardOutput="append:$LOG" -p StandardError="append:$LOG" \
    -E PATH="$PATH" -E HOME="$HOME" -E LANG="${LANG:-C.UTF-8}" \
    -E XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
    -E DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}" \
    -E DISPLAY="$DISPLAY" -E WAYLAND_DISPLAY="$WAYLAND_DISPLAY" -E PULSE_SERVER="$PULSE_SERVER" \
    -E WSL_DISTRO_NAME="${WSL_DISTRO_NAME:-Ubuntu}" -E WSL_INTEROP=/run/WSL/1_interop \
    -E WSL2_GUI_APPS_ENABLED="${WSL2_GUI_APPS_ENABLED:-1}" \
    ${HERMES_HOME:+-E HERMES_HOME="$HERMES_HOME"} \
    "$@"
}

stop_session() {
  local pid
  pid="$(app_pid)"
  if [ -n "$pid" ]; then
    # SIGTERM runs Electron's normal quit teardown (backend child included);
    # concurrently -k then stops the paired vite server.
    log "stop: SIGTERM electron pid $pid"
    kill -TERM "$pid" 2>/dev/null
  fi
  for _ in $(seq 1 40); do
    [ -z "$(app_pid)" ] && ! unit_active && ! port_busy && return 0
    sleep 0.5
  done
  if unit_active; then
    log "stop: session still up after 20s; stopping unit $UNIT"
    systemctl --user stop "$UNIT"
  fi
  [ -z "$(app_pid)" ] && ! port_busy && return 0
  log "stop: could not stop session (electron pid: $(app_pid), port 5174 busy: $(port_busy && echo yes || echo no))"
  return 1
}

case "${1:-}" in
  --stop)
    stop_session
    exit $?
    ;;
  --restart)
    stop_session || exit 1
    ;;
  "") ;;
  *)
    echo "usage: $0 [--restart|--stop]" >&2
    exit 2
    ;;
esac

cd "$DESKTOP_DIR" || exit 1

if [ -n "$(app_pid)" ]; then
  # Running app: this extra Electron hands off to it (restore + show + focus)
  # and exits within a few seconds, so it can run in the foreground.
  log "app already running (pid $(app_pid)); focusing its window"
  electron_env timeout 30 "$ELECTRON_BIN" . >>"$LOG" 2>&1 </dev/null
  exit 0
fi

if unit_active; then
  log "dev session is still starting (unit $UNIT active, no window yet); not starting another"
  exit 0
fi

if dev_server_up; then
  # A dev server without an app (e.g. vite started by hand): just add Electron.
  log "dev server already up without an app; starting electron as unit $UNIT"
  start_unit env XCURSOR_SIZE=24 HERMES_DESKTOP_DEV_SERVER="$DEV_URL" "$ELECTRON_BIN" .
  exit $?
fi

if port_busy; then
  log "port 5174 is held by something that is not the vite dev server; not starting"
  echo "Port 5174 is in use by another program; see $LOG" >&2
  exit 1
fi

log "starting dev session (npm run dev) as unit $UNIT"
start_unit npm run dev
